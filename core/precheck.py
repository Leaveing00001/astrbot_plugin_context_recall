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

#: 审查力度 0~3，数字越大判得越严（0 只拦胡言乱语，3 连空话和脑补都拦）
STRICTNESS_MIN = 0
STRICTNESS_MAX = 3
STRICTNESS_DEFAULT = 1

_JUDGE_BASE = """你是审校员，只判断机器人即将发出的回复能不能发。

{persona}

{criteria}

另外：对方问「你是谁 / 你叫什么 / 你是干嘛的」时，回复按上面的人格设定介绍自己，
不算答非所问，也不算编造身份。

输出格式（严格遵守，不许有多余内容）：
- 通过：输出 `OK`
- 不通过：输出 `BAD|不超过10个字的理由`
不要解释，不要换行，不要加任何其它文字。"""

#: 人格设定最多塞多少字进审查提示词（太长会拖慢 prefill，而且审查只需要认出"我是谁"）
PERSONA_MAX_CHARS = 700


def persona_block(persona: str = "") -> str:
    """把人格设定整理成审查/重写用的提示词片段。"""
    text = (persona or "").strip()
    if not text:
        return "机器人没有额外的人格设定；它的自述只要在上下文里不自相矛盾，就不算编造。"
    if len(text) > PERSONA_MAX_CHARS:
        text = text[: PERSONA_MAX_CHARS - 1].rstrip() + "…"
    return f"机器人的人格设定（与这里一致的内容一律算事实，不算编造）：\n{text}"

#: 四档判据。0 最松、3 最严，措辞刻意写短（每次审查只多几十个 token）
_STRICTNESS_CRITERIA: dict[int, str] = {
    0: """只有这两种才算不通过：
1. 自相矛盾：句内前后冲突，或与上文刚确认的事实冲突；
2. 语无伦次：句子断裂，读不出完整意思。

除此之外一律算通过：答非所问、话题跳了、说得平淡、有点啰嗦、观点不同，全都放行。""",
    1: """只有这四种才算不通过：
1. 答非所问：对方问 A，回复在讲 B；
2. 自相矛盾：句内前后冲突，或与上文刚确认的事实冲突；
3. 事实错乱：编造与上文直接冲突的事实；
4. 语无伦次：句子断裂，读不出完整意思。

以下都必须算通过：观点不同、语气不好、不够有趣、有点啰嗦、答得不够全面。""",
    2: """下面这些都要判不通过：
1. 答非所问：对方问 A，回复在讲 B；
2. 自相矛盾 / 事实错乱：跟上文刚确认的事实冲突，或凭空编造事实；
3. 语无伦次：句子断裂，读不出完整意思；
4. 接错对象：最后那条消息明确 @ 了别人（不是机器人自己）、或者是在问第三个人，
   它却接过来回答——这是抢别人的话；
5. 话题错位：回复和最近几条完全不在一个话题上，凭空冒出上下文里没有的名词。

以下仍然算通过：观点不同、语气不好、不够有趣、有点啰嗦。""",
    3: """下面这些都要判不通过：
1. 答非所问：对方问 A，回复在讲 B；
2. 自相矛盾 / 事实错乱：跟上文刚确认的事实冲突，或凭空编造事实；
3. 语无伦次：句子断裂，读不出完整意思；
4. 接错对象：最后那条消息明确 @ 了别人（不是机器人自己）、或者是在问第三个人，
   它却接过来回答；
5. 话题错位：回复和最近几条完全不在一个话题上；
6. 空话：只有语气词、情绪或敷衍（「嗯」「哈哈」「好家伙」），没有任何信息；
7. 重复：和它自己刚说过的话意思几乎一样；
8. 脑补：说起上下文里根本没出现过的人、事、地点、经历。

只有「贴题、有信息、不编造」才算通过。""",
}


def clamp_strictness(value: object) -> int:
    """把审查力度夹在 0~3，非数字回落默认值。"""
    try:
        level = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return STRICTNESS_DEFAULT
    return max(STRICTNESS_MIN, min(STRICTNESS_MAX, level))


def judge_system(strictness: int = STRICTNESS_DEFAULT, persona: str = "") -> str:
    """按力度 + 人格设定生成裁决用的 system prompt。"""
    return _JUDGE_BASE.format(
        persona=persona_block(persona),
        criteria=_STRICTNESS_CRITERIA[clamp_strictness(strictness)],
    )


#: 默认力度（1 = 标准）的 system prompt，保留这个名字给老代码用
JUDGE_SYSTEM = judge_system(STRICTNESS_DEFAULT)


_REWRITE_BASE = """你正在把自己刚说错的一句话重写。
{persona}
要求：
- 只修正被指出的问题，语气、人称、长度尽量保持原样；
- 直接输出重写后的消息本身，不要引号、不要前缀、不要解释、不要换行。

只输出这条消息。"""


def rewrite_system(persona: str = "") -> str:
    """重写用的 system prompt；带人格设定，免得改完不像同一个人。"""
    text = (persona or "").strip()
    block = f"\n{persona_block(text)}\n" if text else "\n"
    return _REWRITE_BASE.format(persona=block)


#: 不带人格设定的重写 prompt（老代码兼容）
REWRITE_SYSTEM = rewrite_system()


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
