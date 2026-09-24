import json

from jev_windows import llm_judge
from jev_windows.models import ChatSnapshot, Message

SNAPSHOT = ChatSnapshot("朋友", [Message("her", "你又忘了吧"), Message("me", "忘了什么")])

BASE = {
    "literal_question": False,
    "true_intent": "confirm_you_care",
    "danger_level": 5,
    "should_reply_now": 0.2,
    "best_action": "check_history",
    "she_needs": "care",
    "tension_resolved": 0.1,
}


def _reply(monkeypatch, payload):
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    monkeypatch.setattr(
        llm_judge,
        "post_chat",
        lambda *args, **kwargs: {"choices": [{"message": {"content": content}}]},
    )


def test_topic_hooks_are_deduplicated_and_trimmed(monkeypatch):
    _reply(monkeypatch, {**BASE, "topic_hooks": ["加班太多没回消息", " 你说的那家店 ", "加班太多没回消息"]})
    analysis = llm_judge.judge_with_llm(SNAPSHOT, "朋友", "key")
    assert analysis.topic_hooks == ["加班太多没回消息", "你说的那家店"]


def test_topic_hooks_accept_comma_separated_string(monkeypatch):
    _reply(monkeypatch, {**BASE, "topic_hooks": "周末吃饭、看电影,打球"})
    analysis = llm_judge.judge_with_llm(SNAPSHOT, "朋友", "key")
    assert analysis.topic_hooks == ["周末吃饭", "看电影", "打球"]


def test_overlong_or_missing_hooks_are_dropped(monkeypatch):
    _reply(monkeypatch, {**BASE, "topic_hooks": ["这条非常非常非常非常非常非常非常长一定会被丢掉", "", None]})
    analysis = llm_judge.judge_with_llm(SNAPSHOT, "朋友", "key")
    assert analysis.topic_hooks == []


def test_hooks_default_to_empty_list_when_absent(monkeypatch):
    _reply(monkeypatch, BASE)
    analysis = llm_judge.judge_with_llm(SNAPSHOT, "朋友", "key")
    assert analysis.topic_hooks == []


def test_probability_and_score_are_normalized(monkeypatch):
    _reply(
        monkeypatch,
        {
            **BASE,
            "danger_level": 12,
            "should_reply_now": 85,
            "tension_resolved": "true",
            "true_intent": "anger",
        },
    )
    analysis = llm_judge.judge_with_llm(SNAPSHOT, "朋友", "key")
    assert analysis.danger_level == 9.0
    assert analysis.should_reply_now == 0.85
    assert analysis.tension_resolved == 1.0
    assert analysis.true_intent == "vent_anger"


def test_array_payload_is_unwrapped_instead_of_crashing(monkeypatch):
    """模型偶尔外层给数组包一层，要能抠出里面的对象而不是崩掉。"""
    _reply(monkeypatch, '[{"true_intent": "casual_chat", "danger_level": 1}]')
    analysis = llm_judge.judge_with_llm(SNAPSHOT, "朋友", "key")
    assert analysis.true_intent == "casual_chat"
    assert analysis.danger_level == 1.0
