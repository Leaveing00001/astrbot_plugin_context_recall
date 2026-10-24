"""通过 OneBot v11 接口拉取最近的聊天记录。

只依赖一个「有 async call_action 方法的对象」，因此也能脱离 astrbot 测试。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class HistoryUnavailable(RuntimeError):
    """当前协议端不支持或拒绝提供聊天记录。"""


def _as_int(value: str) -> int | str:
    text = str(value or "").strip()
    return int(text) if text.isdigit() else text


async def fetch_history(
    bot: Any,
    *,
    is_group: bool,
    peer_id: str,
    count: int,
) -> list[dict]:
    """拉取某个会话最近的 ``count`` 条消息（旧 → 新）。

    Raises:
        HistoryUnavailable: 协议端不支持该接口，或返回了非预期结构。
    """
    if bot is None or not hasattr(bot, "call_action"):
        raise HistoryUnavailable("当前平台不支持 call_action，无法读取聊天记录")

    action = "get_group_msg_history" if is_group else "get_friend_msg_history"
    key = "group_id" if is_group else "user_id"
    payload: dict[str, Any] = {key: _as_int(peer_id), "count": max(1, int(count))}

    try:
        data = await bot.call_action(action, **payload)
    except HistoryUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - 协议端异常形态不可控
        raise HistoryUnavailable(f"{action} 调用失败: {exc}") from exc

    messages: object = None
    if isinstance(data, dict):
        messages = data.get("messages")
        if messages is None:
            messages = data.get("message")
    elif isinstance(data, list):
        messages = data
    if not isinstance(messages, list):
        raise HistoryUnavailable(f"{action} 返回结构异常: {type(data).__name__}")
    return [item for item in messages if isinstance(item, dict)]
