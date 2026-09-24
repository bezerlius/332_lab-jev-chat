"""DeepSeek 版判断引擎：用普通大模型顶替 Jev 的结构化判断。

对外只暴露 ``judge_with_llm()``，返回与 ``jev_api.judge()`` 完全相同的
``Analysis`` 结构，因此 app.py / deepseek_client.py 等下游代码无需改动。

题目定义仍然复用 ``tools/jev/questions.py`` 里的 ``JUDGE_QUESTIONS``，
避免维护两份语义不一致的判断题。
"""

from __future__ import annotations

import json
import re
import time

from tools.jev.questions import JUDGE_QUESTIONS

from .deepseek_client import post_chat
from .models import Analysis, ChatSnapshot


DEFAULT_JUDGE_MODEL = "deepseek-chat"

# 合法选项直接从题目定义里派生，改题目时不用再改这里。
INTENT_CHOICES = tuple(JUDGE_QUESTIONS["true_intent"]["criteria"].keys())
ACTION_CHOICES = tuple(JUDGE_QUESTIONS["best_action"]["criteria"].keys())
NEED_CHOICES = tuple(JUDGE_QUESTIONS["she_needs"]["criteria"].keys())

# 归一化兜底：模型偶尔会输出同义写法。
_INTENT_ALIASES = {
    "confirm_care": "confirm_you_care",
    "test_you_care": "confirm_you_care",
    "vent": "vent_anger",
    "anger": "vent_anger",
    "request": "request_action",
    "ask_action": "request_action",
    "explanation": "seek_explanation",
    "seek_explain": "seek_explanation",
    "casual": "casual_chat",
    "chat": "casual_chat",
    "close": "close_topic",
    "closed": "close_topic",
}
_ACTION_ALIASES = {
    "checkhistory": "check_history",
    "check_chat": "check_history",
    "apologise": "apologize",
    "commitment": "give_commitment",
    "promise": "give_commitment",
    "acknowledge_feeling": "acknowledge",
    "less": "say_less",
    "plan": "make_plan",
}
_NEED_ALIASES = {
    "apologize": "apology",
    "apologise": "apology",
    "act": "action",
    "explain": "explanation",
    "be_cared": "care",
    "care_attention": "care",
    "none": "nothing",
    "no": "nothing",
}

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class LlmJudgeError(RuntimeError):
    pass


def _criteria_lines(spec: dict) -> list[str]:
    criteria = spec.get("criteria")
    lines: list[str] = []
    if isinstance(criteria, dict):
        for key, description in criteria.items():
            lines.append(f"      - {key}: {description}")
    elif isinstance(criteria, list):
        for index, description in enumerate(criteria):
            lines.append(f"      - {index}: {description}")
    return lines


def build_question_prompt() -> str:
    """把 JUDGE_QUESTIONS 渲染成给大模型的英文题面（题面本身保持原样）。"""
    blocks: list[str] = []
    for name, spec in JUDGE_QUESTIONS.items():
        lines = [f"  - {name} (type: {spec.get('type')})"]
        instructions = spec.get("instructions")
        if instructions:
            lines.append(f"      instructions: {instructions}")
        lines.extend(_criteria_lines(spec))
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


SYSTEM_PROMPT = (
    "You are a calibrated relationship-judgment engine. "
    "You answer a fixed set of judgment questions about a Chinese chat transcript. "
    "Judge the LATEST message in the context of the WHOLE thread, not one line in isolation. "
    "Prefer tone, subtext and context over surface wording. "
    "Never invent facts that are not in the transcript. "
    "Reply with JSON only, no markdown fence, no commentary."
)

OUTPUT_CONTRACT = (
    "Return exactly one JSON object with these keys:\n"
    "{\n"
    '  "literal_question": true or false,\n'
    f'  "true_intent": one of {list(INTENT_CHOICES)},\n'
    '  "danger_level": integer 0-9,\n'
    '  "should_reply_now": number between 0 and 1 (your confidence that substantive content is warranted),\n'
    f'  "best_action": one of {list(ACTION_CHOICES)},\n'
    f'  "she_needs": one of {list(NEED_CHOICES)},\n'
    '  "tension_resolved": number between 0 and 1,\n'
    '  "topic_hooks": array of 2-3 strings\n'
    "}\n"
    "Rules:\n"
    "- Use numbers, not strings, for danger_level / should_reply_now / tension_resolved.\n"
    "- topic_hooks: directions the user could naturally steer the chat toward next. "
    "Each hook must come from something actually said or clearly implied in the transcript "
    "(a mentioned event, person, taste, plan, worry, detail). "
    "Each hook is at most 12 Chinese characters, written as a direction not a full sentence. "
    "Never invent things that never came up. If the thread gives nothing usable, return an empty array."
)


def _transcript(snapshot: ChatSnapshot) -> str:
    return "\n".join(
        f"{'me' if message.side == 'me' else 'her'}: {message.text}"
        for message in snapshot.messages[-10:]
    )


def _extract_json(text: str) -> dict:
    """模型偶尔会裹 ```json、加前后缀，甚至返回数组，这里做一层稳健提取。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    candidates = [text]
    match = _JSON_OBJECT_RE.search(text)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise LlmJudgeError("DeepSeek 判断没有返回可解析的 JSON 对象。")


def _normalize_choice(value, allowed: tuple[str, ...], aliases: dict[str, str], fallback: str) -> str:
    if value is None:
        return fallback
    raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return fallback
    if raw in allowed:
        return raw
    mapped = aliases.get(raw)
    if mapped in allowed:
        return mapped
    for choice in allowed:
        if choice in raw or raw in choice:
            return choice
    return fallback


def _normalize_probability(value) -> float | None:
    """把布尔 / 0-1 小数 / 0-100 整数 / 中英文词 统一收敛到 0.0-1.0。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 1.0 and number <= 100.0:
            number = number / 100.0
        return min(1.0, max(0.0, number))
    text = str(value).strip().lower()
    if text in {"true", "yes", "是", "高", "true."}:
        return 1.0
    if text in {"false", "no", "否", "低"}:
        return 0.0
    try:
        return _normalize_probability(float(text))
    except ValueError:
        return None


def _normalize_score(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(min(9.0, max(0.0, float(value))), 1)
    text = str(value).strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return round(min(9.0, max(0.0, float(match.group(0)))), 1)


def _string_list(value, limit: int = 3, max_len: int = 16) -> list[str]:
    """把模型给的话题钩子收敛成干净的短字符串列表。"""
    if isinstance(value, str):
        items = re.split(r"[、,，;；\n]+", value)
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return []
    cleaned: list[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip().strip("。.！!\"'“”")
        # 超长说明模型没按要求写短钩子，宁可丢掉也不要塞进 UI 里撑爆布局。
        if not text or len(text) > max_len:
            continue
        if text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def judge_with_llm(
    snapshot: ChatSnapshot,
    relationship: str,
    key: str,
    model: str = DEFAULT_JUDGE_MODEL,
) -> Analysis:
    """用 DeepSeek 完成 7 道判断题，返回与 Jev 一致的 Analysis。"""
    if not key:
        raise LlmJudgeError("缺少 DeepSeek API 密钥。")
    start = time.monotonic()
    model_name = (model or DEFAULT_JUDGE_MODEL).strip()
    is_reasoner = "reasoner" in model_name

    body: dict = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"relationship: {relationship}\n"
                    f"transcript (from=me is the user, from=her is the other person):\n"
                    f"{_transcript(snapshot)}\n\n"
                    "judgment questions:\n"
                    f"{build_question_prompt()}\n\n"
                    f"{OUTPUT_CONTRACT}"
                ),
            },
        ],
        "max_tokens": 800,
        "stream": False,
    }
    if not is_reasoner:
        # reasoner 系列不支持这两个参数，带上会 400。
        body["thinking"] = {"type": "disabled"}
        body["response_format"] = {"type": "json_object"}

    try:
        response = post_chat(key, body, timeout=45)
    except Exception as exc:  # DeepSeekError 及其他网络异常统一向上抛
        raise LlmJudgeError(str(exc)) from exc

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmJudgeError("DeepSeek 判断返回内容为空。") from exc

    answers = _extract_json(content)
    return Analysis(
        true_intent=_normalize_choice(
            answers.get("true_intent"), INTENT_CHOICES, _INTENT_ALIASES, "casual_chat"
        ),
        danger_level=_normalize_score(answers.get("danger_level")),
        need=_normalize_choice(answers.get("she_needs"), NEED_CHOICES, _NEED_ALIASES, "care"),
        best_action=_normalize_choice(
            answers.get("best_action"), ACTION_CHOICES, _ACTION_ALIASES, "acknowledge"
        ),
        should_reply_now=_normalize_probability(answers.get("should_reply_now")),
        tension_resolved=_normalize_probability(answers.get("tension_resolved")),
        topic_hooks=_string_list(answers.get("topic_hooks")),
        latency_ms=int((time.monotonic() - start) * 1000),
    )
