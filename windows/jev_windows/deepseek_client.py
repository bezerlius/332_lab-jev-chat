from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request

from .models import Analysis, ChatSnapshot


CHAT_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-flash"


class DeepSeekError(RuntimeError):
    pass


# 三条线路的固定分工：第 1 条接情绪、第 2 条给抓手、第 3 条抛话题。
# UI 直接按索引贴标签，所以顺序不能变。
REPLY_TACTICS = ("接住情绪", "给出抓手", "抛新话题")

SYSTEM_PROMPT = (
    "你在帮一个真人回微信私聊。要给出三条可以直接复制发出去的候选回复。\n\n"
    "【输出格式】\n"
    "只输出 JSON：{\"replies\":[\"第一条\",\"第二条\",\"第三条\"]}。不要任何解释、标题或序号。\n\n"
    "【三条的分工：必须走不同路数，不许换个词重复同一件事】\n"
    "1) 接住情绪：最短最稳的一句（12-25 字）。承认对方此刻的感受或处境，"
    "不辩解、不讲道理、不展开、道歉不超过一次。\n"
    "2) 给出抓手：带一点具体东西的一句（15-35 字）。一个事实、一个可选时间、一个做法、"
    "一句我这边的情况。让对话能往前挪一步。\n"
    "3) 抛新话题：把对话带到一个新方向（15-35 字）。只能是下面四种之一：\n"
    "   · 追问对方刚提到的某个具体细节\n"
    "   · 一个相关的提议或轻量邀约\n"
    "   · 一句自嘲或吐槽，开启新话题\n"
    "   · 一个跟对方有关的新念头\n"
    "   硬性禁止：不许继续安抚情绪，不许再道歉，不许重复前两条说过的事。\n"
    "   自检：三条如果看着像同一句话的三个版本，就重写。\n\n"
    "【说话方式：说人话】\n"
    "- 像在手机上随手打字：可以断句、省略主语、带语气词。\n"
    "- 一次只说一件事，不要小作文，不要排比句，不要书面语。\n"
    "- 禁止公文腔：首先/其次/总之/希望你能够理解/我会注意的。\n"
    "- 禁止读心术：不要写“我知道你其实…”“你嘴上说没事但其实…”。"
    "可以讲自己的判断，不许替对方宣布感受。\n"
    "- 禁止空洞自责循环：道歉最多一次，后面必须接动作或话题；不要连用“对不起我错了”。\n"
    "- 禁止编造：不捏造没发生过的记忆、日期、承诺、事件。不知道就诚实说不知道。\n"
    "- 不要煽情表白，不要写“我保证以后…”这类兑现不了的话。\n\n"
    "【幽默分寸：看危险程度】\n"
    "- 危险 0-3：尽情皮。自嘲、吐槽、夸张比喻、轻松调侃都可以，"
    "别端着，别把天聊成慰问现场。\n"
    "- 危险 4-6：温和调侃，不许抖机灵。\n"
    "- 危险 7-9：完全不许开玩笑、反讽、转移话题，只许稳、短、诚恳。\n\n"
    "【示例：体会三条的分工差异】\n"
    "对话：对方说“今天跑步差点把我累死 / 三公里就不行了”\n"
    "输出：{\"replies\":["
    "\"三公里还行啊，比上次强。\","
    "\"明天要不要一起去，我陪你慢慢跑。\","
    "\"话说你是突然想起来要锻炼，还是被谁刺激了？\""
    "]}\n\n"
    "【其余】\n"
    "- 贴合给定关系调整称呼与亲密度。\n"
    "- 不要替用户发送，不要在回复里加括号舞台说明。"
)


def _post(key: str, body: dict, timeout: float = 35) -> dict:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(2):
        request = urllib.request.Request(
            CHAT_URL,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            status = exc.code
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            if status in (429, 500, 502, 503) and attempt == 0:
                time.sleep(1)
                continue
            readable = {
                401: "DeepSeek API 密钥无效（401）",
                402: "DeepSeek API 余额不足（402）",
                429: "DeepSeek 请求过于频繁（429）",
            }.get(status, f"DeepSeek 请求失败（HTTP {status}）：{detail}")
            raise DeepSeekError(readable) from None
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(1)
                continue
    raise DeepSeekError(f"无法连接 DeepSeek API：{last_error}")


def post_chat(key: str, body: dict, timeout: float = 35) -> dict:
    """公开的 DeepSeek /chat/completions 请求入口，供 llm_judge 等模块复用。

    内部转发到 ``_post``，这样测试里 monkeypatch ``_post`` 对本函数同样生效。
    """
    return _post(key, body, timeout)


def generate_suggestions(
    snapshot: ChatSnapshot,
    relationship: str,
    analysis: Analysis,
    key: str,
    model: str = DEFAULT_MODEL,
) -> list[str]:
    transcript = "\n".join(
        f"{'我' if message.side == 'me' else '对方'}：{message.text}"
        for message in snapshot.messages
    )
    # 判断引擎不可用时字段会是空/None，这里统一标注为"未提供"，
    # 让生成模型自己从对话推断，而不是拿到一堆 null。
    missing = "未提供"
    judgment = {
        "true_intent": analysis.true_intent or missing,
        "danger_level_0_to_9": analysis.danger_level if analysis.danger_level is not None else missing,
        "need": analysis.need or missing,
        "best_action": analysis.best_action or missing,
        "should_reply_probability": (
            analysis.should_reply_now if analysis.should_reply_now is not None else missing
        ),
        "tension_resolved_probability": (
            analysis.tension_resolved if analysis.tension_resolved is not None else missing
        ),
    }
    hooks = [str(hook).strip() for hook in (analysis.topic_hooks or []) if str(hook).strip()]
    hooks_line = (
        "可聊话题钩子（从对方说过的话里挖的，第三条优先拿它当素材）：" + " / ".join(hooks)
        if hooks
        else "可聊话题钩子：判断环节没给，请自己从对话里找一个对方真实提过的人、事或细节当钩子"
    )
    response = _post(
        key,
        {
            "model": model or DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"关系：{relationship}\n"
                        f"结构化判断：{json.dumps(judgment, ensure_ascii=False)}\n"
                        f"{hooks_line}\n"
                        f"对话：\n{transcript}"
                    ),
                },
            ],
            "thinking": {"type": "disabled"},
            "max_tokens": 600,
            # 让三条回复尽量走不同路子，少出现换词重复
            "presence_penalty": 0.5,
            "temperature": 1.0,
            "response_format": {"type": "json_object"},
            "stream": False,
        },
    )
    try:
        content = response["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        replies = parsed["replies"]
        cleaned = []
        for item in replies:
            # 模型偶尔会自作主张返回 {"text": "..."} 结构，这里兜住。
            if isinstance(item, dict):
                item = item.get("text") or item.get("reply") or ""
            text = str(item).strip()
            if text:
                cleaned.append(text)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DeepSeekError("DeepSeek 没有返回有效的建议回复。") from exc
    if len(cleaned) != 3:
        raise DeepSeekError("DeepSeek 返回的建议回复不是三条。")
    return cleaned

