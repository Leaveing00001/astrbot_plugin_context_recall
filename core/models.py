"""把 OneBot 的原始聊天记录规范成一行行可读文本。

这一层只做纯数据处理，不依赖 astrbot，方便单独测试。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CQ_CODE_RE = re.compile(r"\[CQ:[^\]]*\]")

#: 非文本消息段的可读替代
_SEGMENT_LABELS = {
    "image": "[图片]",
    "face": "[表情]",
    "mface": "[表情]",
    "sticker": "[表情]",
    "record": "[语音]",
    "video": "[视频]",
    "file": "[文件]",
    "forward": "[聊天记录]",
    "node": "[聊天记录]",
    "json": "[卡片]",
    "xml": "[卡片]",
    "markdown": "[卡片]",
    "share": "[链接]",
    "music": "[音乐]",
    "location": "[位置]",
    "dice": "[骰子]",
    "rps": "[猜拳]",
    "poke": "[戳一戳]",
}

#: 这些消息段对语义判断没有帮助，直接丢掉
_IGNORED_SEGMENTS = {"reply", "source"}


def _segment_data(segment: dict) -> dict:
    data = segment.get("data")
    return data if isinstance(data, dict) else {}


def segment_to_text(segment: object) -> str:
    """把一个消息段转成一行里的可读片段。"""
    if not isinstance(segment, dict):
        return ""
    kind = str(segment.get("type") or "")
    data = _segment_data(segment)
    if kind == "text":
        return str(data.get("text") or "")
    if kind == "at":
        qq = str(data.get("qq") or "")
        return "" if qq in {"", "all"} else f"@{qq}"
    if kind in _IGNORED_SEGMENTS:
        return ""
    if kind in _SEGMENT_LABELS:
        return _SEGMENT_LABELS[kind]
    return f"[{kind}]" if kind else ""


def truncate(text: str, limit: int) -> str:
    """按字符截断，超长时补省略号。"""
    if limit and limit > 0 and len(text) > limit:
        return text[: max(1, limit - 1)] + "…"
    return text


def message_to_text(raw: dict, limit: int = 220) -> str:
    """从一条原始消息里挤出纯文本。"""
    fragments: list[str] = []
    segments = raw.get("message")
    if isinstance(segments, list):
        for segment in segments:
            fragment = segment_to_text(segment)
            if fragment:
                fragments.append(fragment)
    if not fragments:
        fallback = raw.get("raw_message") or raw.get("message_str") or ""
        fallback = _CQ_CODE_RE.sub("", str(fallback))
        fragments = [fallback]
    text = " ".join(" ".join(fragments).split())
    return truncate(text, limit)


@dataclass(frozen=True, slots=True)
class ChatLine:
    """聊天记录里的一行。"""

    message_id: str
    user_id: str
    name: str
    ts: float
    text: str
    is_self: bool = False

    def speaker(self) -> str:
        return "我（机器人）" if self.is_self else (self.name or "某人")


def _coerce_ts(value: object) -> float:
    try:
        ts = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if ts > 1e12:  # 毫秒时间戳
        ts /= 1000.0
    return ts


def normalize_messages(
    raw_messages: object,
    self_id: str,
    limit: int = 220,
) -> list[ChatLine]:
    """把 ``get_*_msg_history`` 返回的 messages 列表规范化并排序。"""
    if not isinstance(raw_messages, list):
        return []
    lines: list[ChatLine] = []
    self_id = str(self_id or "")
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        sender = raw.get("sender")
        sender = sender if isinstance(sender, dict) else {}
        user_id = str(
            sender.get("user_id") or raw.get("user_id") or raw.get("sender_id") or ""
        )
        name = str(sender.get("card") or sender.get("nickname") or user_id or "?")
        ts = _coerce_ts(raw.get("time") or raw.get("timestamp"))
        message_id = str(raw.get("message_id") or raw.get("msg_id") or "")
        text = message_to_text(raw, limit)
        if not message_id and not text:
            continue
        lines.append(
            ChatLine(
                message_id=message_id,
                user_id=user_id,
                name=name,
                ts=ts,
                text=text,
                is_self=bool(self_id) and user_id == self_id,
            )
        )
    lines.sort(key=lambda line: line.ts)
    return lines
