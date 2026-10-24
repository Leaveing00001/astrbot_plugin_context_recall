"""发送前的逻辑审查：看一眼、不合格就打回重写，重写的还会再审一遍。

设计目标：**快**。
- 裁决只要求模型吐 `OK` 或 `BAD|理由`，输出约 5 个 token；
- 重写只要求吐消息本身，不解释、不加引号；
- 正常消息只花一次调用，出问题才有第二次、第三次。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .judge import matches_any, truncate_reason

# ---------------------------------------------------------------------------
# 提示词（刻意写短：输入短、输出更短）
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """你是审校员，只判断机器人即将发出的回复是否明显不合逻辑。

只有这四种才算：
1. 答非所问：对方问 A，回复在讲 B；
2. 自相矛盾：句内前后冲突，或与上文刚确认的事实冲突；
3. 事实错乱：编造与上文直接冲突的事实；
4. 语无伦次：句子断裂，读不出完整意思。

以下都必须算通过：观点不同、语气不好、不够有趣、有点啰嗦、答得不够全面。

输出格式（严格遵守，不许有多余内容）：
- 通过：输出 `OK`
- 不通过：输出 `BAD|不超过10个字的理由`
不要解释，不要换行，不要加任何其它文字。"""


REWRITE_SYSTEM = """你正在把自己刚说错的一句话重写。

要求：
- 只修正被指出的问题，语气、人称、长度尽量保持原样；
- 直接输出重写后的消息本身，不要引号、不要前缀、不要解释、不要换行。

只输出这条消息。"""


DEFAULT_JUDGE_PROMPT = """【场景】{{session_label}}

【最近聊天】
{{transcript}}

【待审回复】
{{draft}}

这条回复是否明显不合逻辑？"""


DEFAULT_REWRITE_PROMPT = """【场景】{{session_label}}

【最近聊天】
{{transcript}}

【你刚才的回复】{{draft}}

【问题】{{reason}}

请重写这条回复，改掉上面的问题。只输出重写后的消息本身。"""


# ---------------------------------------------------------------------------
# 工具调用 / 结构化消息：出现就整条放行，不审也不改
# ---------------------------------------------------------------------------

DEFAULT_TOOL_PATTERNS: tuple[str, ...] = (
    # 工具调用的 XML / 特殊 token / JSON 痕迹
    r"<\s*/?\s*(?:tool|tools|function_call|function_calls|invoke|parameter)\b",
    r"<\|[a-z_]{2,20}\|>",
    r"(?i)\b(?:tool_calls|tool_call_id|tool_use|tool_result|function_call|function_calls)\b",
    r'(?i)"type"\s*:\s*"(?:tool_use|tool_result|function)"',
    r'(?i)"arguments"\s*:\s*\{',
    r"(?i)\bAction\s*Input\s*:|\bObservation\s*:",
    r"【工具调用】|\[TOOL\]|\[工具\]|\[调用工具",
    r"(?i)mcp__|<antml:",
    # 平台的地址簿语法：CQ 码 / [At:123] / [Reply:..] / reply_to / @123456789
    r"\[CQ:[a-zA-Z_]+,",
    r"\[At\s*[:：][^\]]{1,32}\]",
    r"(?i)\[reply\s*[:：][^\]]{0,32}\]",
    r"(?i)\breply_to\b",
    r"@\d{5,12}(?!\d)",
)


@dataclass(frozen=True, slots=True)
class PrecheckVerdict:
    ok: bool
    reason: str = ""


def has_tool_markers(text: str, patterns: Sequence[str] = DEFAULT_TOOL_PATTERNS) -> bool:
    """文本里是否夹带了工具调用 / 特殊 token 痕迹。"""
    return matches_any(text, patterns)


def build_judge_prompt(
    template: str,
    session_label: str,
    transcript: str,
    draft: str,
) -> str:
    return (
        template.replace("{{session_label}}", session_label)
        .replace("{{transcript}}", transcript)
        .replace("{{draft}}", draft)
    )


def build_rewrite_prompt(
    template: str,
    session_label: str,
    transcript: str,
    draft: str,
    reason: str,
) -> str:
    return (
        template.replace("{{session_label}}", session_label)
        .replace("{{transcript}}", transcript)
        .replace("{{draft}}", draft)
        .replace("{{reason}}", reason or "明显不合逻辑")
    )


def parse_verdict_line(raw: str) -> PrecheckVerdict:
    """解析 `OK` / `BAD|理由`。解析不出来一律当通过（宁可放行，不吞消息）。"""
    text = (raw or "").strip()
    if not text:
        return PrecheckVerdict(True)
    first = text.splitlines()[0].strip()
    clean = first.strip("*`'\"“”‘’ \t")
    upper = clean.upper()

    if upper.startswith("OK") or upper.startswith("PASS"):
        return PrecheckVerdict(True)
    if upper.startswith(("BAD", "FAIL", "ABNORMAL")):
        reason = re.sub(r"^[A-Z]+[\s|:：、-]*", "", clean).lstrip("|:：、- ").strip()
        return PrecheckVerdict(False, truncate_reason(reason or "明显不合逻辑", 20))
    if "不合逻辑" in text or "答非所问" in text:
        return PrecheckVerdict(False, truncate_reason(clean, 20))
    if "OK" in upper:
        return PrecheckVerdict(True)
    return PrecheckVerdict(True)


_MARKDOWN_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_PREFIXES = (
    "重写后的消息",
    "重写后",
    "重写的消息",
    "改写后",
    "修正后",
    "回复",
    "答",
    "消息",
    "重写",
    "改写",
)


def clean_rewrite(raw: str, limit: int = 400) -> str:
    """把模型输出的重写结果收拾干净：去代码块、去前缀、去引号。"""
    text = (raw or "").strip()
    if not text:
        return ""
    text = _MARKDOWN_FENCE.sub("", text).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    text = lines[0]
    for prefix in _PREFIXES:
        for sep in ("：", ":", " "):
            head = prefix + sep
            if text.startswith(head):
                text = text[len(head) :].strip()
                break
    text = text.strip().strip("\"“”「」『』'`").strip()
    text = truncate_reason(text, limit)
    return text
