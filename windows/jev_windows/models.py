from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


@dataclass(frozen=True)
class Message:
    side: str
    text: str


@dataclass(frozen=True)
class ChatSnapshot:
    title: str
    messages: list[Message]
    raw_text: str = ""

    def signature(self) -> str:
        return "|".join(f"{m.side}:{m.text}" for m in self.messages[-6:])


@dataclass
class Analysis:
    true_intent: str = ""
    danger_level: float | None = None
    need: str = ""
    best_action: str = ""
    should_reply_now: float | None = None
    tension_resolved: float | None = None
    latency_ms: int = 0
    # 从对话里挖出来的、可以继续聊下去的方向。仅 LLM 判断引擎会填，
    # Jev 后端拿不到时为默认空列表，下游按"没有钩子"处理。
    topic_hooks: list[str] = field(default_factory=list)
