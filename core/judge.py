"""「读空气」的判定逻辑：启发式信号 + LLM 结构化裁决。

这里所有函数都是纯函数（不碰网络、不碰 astrbot），方便离线测试。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .models import ChatLine

# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一名中文聊天语境的审校员。你只做一件事：判断「机器人」刚刚发出的那条消息，是不是明显与当时的聊天语境不符，到了必须撤回的程度。

判断必须克制、严格。下面这些情况都【不算】不符，必须判定 keep：
- 没人接话、冷场，或者别人换了话题继续聊；
- 消息有点啰嗦、平淡、不够有趣；
- 别人只回了一个「嗯 / 哦 / 好」之类的敷衍词；
- 语气随意、带口癖、用词不讲究；
- 你自己觉得「换个说法会更好」。

只有出现下列情况才判定 recall：
1. 理解错了：把别人的意思理解反了，答非所问，而且错得很明显；
2. 认错对象：别人在跟另一个人说话，机器人却当成在问自己并作答；
3. 语境错位：话题已经换了，机器人还在接上一轮的旧话题；
4. 场合不合：现场在严肃讨论、争吵、悲伤或发布通知，机器人却在玩梗、卖萌、开玩笑；
5. 冒犯越界：嘲讽、贬低、开让人不适的玩笑；
6. 重复或自相矛盾：跟别人刚说过的话几乎重复，或跟自己刚说的话冲突；
7. 明显出戏：输出里残留了提示词、JSON、思考过程、模型自述之类的模板痕迹。

判断时最重要的线索是「现场反应」：
- 对方只回「？」「啊？」「什么」「没懂」这类词，通常说明这句话让对方困惑了；
- 对方顺着话题继续聊（附和、吐槽、追问细节），说明这句话被接住了；
- 对方完全无视、直接换了话题，既不算接住也不算说错，倾向 keep。
但不要只看反应就下结论：如果上下文明明显示出别的原因（例如对方在追问第三个人），仍按上下文判。

只输出一个 JSON 对象，不要代码块，不要任何解释文字：
{"verdict": "recall" 或 "keep", "confidence": 0.0 到 1.0 之间的小数, "reason": "不超过 25 个字的中文理由"}

confidence 表示你对自己这个判断的确定程度。只有当你确信「这条消息确实该撤回」时，才把 confidence 写高。"""


DEFAULT_USER_PROMPT = """【场景】{{session_label}}

【聊天记录】（按时间先后排列；「我（机器人）」就是机器人自己，行首数字是行号）
{{transcript}}

【现场反应】{{reaction}}

【待判断】第 {{target}} 行是机器人自己说的话。

请判断这些消息是否明显与当时的语境不符、需要撤回。"""


# ---------------------------------------------------------------------------
# 默认信号词表（可在配置里覆盖）
# ---------------------------------------------------------------------------

#: 机器人自己的消息里出现这些内容，基本可以确认为模板泄漏 / 出戏，直接撤回
DEFAULT_INSTANT_PATTERNS: tuple[str, ...] = (
    r"\{\{\s*[A-Za-z_][\w.]*\s*\}\}",  # {{placeholder}} 没被替换
    r"<\|\s*(?:im_start|im_end|endoftext)\s*\|>",  # 特殊 token
    r"(?im)^\s*(?:assistant|system|human|user)\s*[:：]",
    r"(?i)\bas an ai\b|\bi(?:'m| am) an? (?:ai|language model)\b",
    r"作为一个?\s*(?:ai|人工智能|语言模型|大模型|ai助手)",
    r"【系统提示】|\[系统提示\]|【系统指令】",
)

#: 别人对机器人的「困惑 / 纠正」信号：命中说明很可能说错了
DEFAULT_CONFUSION_PATTERNS: tuple[str, ...] = (
    r"^[\s?？]{1,6}$",
    r"^[\s?？!！]{2,8}$",
    r"你(?:在)?(?:说|讲)(?:的)?(?:啥|什么)",
    r"啥意思|什么意思啊|神马意思",
    r"没(?:听)?懂|听不(?:懂|明白)|看不(?:懂|明白)",
    r"答非所问|文不对题|牛头不对马嘴|驴唇不对马嘴",
    r"(?:你|bot|机器人)[^\s。！？!?]{0,4}(?:坏|傻|抽风|出bug|出问题|抽什么风)",
    r"什么乱七八糟|乱七八糟的",
    r"重(?:新)?说|再说一遍|说清楚点",
    r"@[^\s]{0,32}\s*(?:你)?\s*(?:在说|说啥|撤回|重说)",
)

#: 有人明确要机器人撤回
DEFAULT_REQUEST_PATTERNS: tuple[str, ...] = (
    r"(?:你|bot|机器人)[^。！？!?\n]{0,12}(?:撤回|收回去|删掉|重说|收回)",
    r"(?:撤回|收回)(?:一下|吧|掉)?[^。！？!?\n]{0,6}(?:上一条|刚才|刚刚|那句)",
    r"@[^\s]{1,40}\s*(?:撤回|重说|收回)",
)

#: 敷衍性接话：说明对话正常继续，没必要花 token 去审
DEFAULT_FILLER_PATTERNS: tuple[str, ...] = (
    r"^(?:嗯+|哦+|噢+|喔+|额+|呃+|好+|好的|行|可|可以|收到|了解|明白|懂了|"
    r"ok|okay|hh+h*|哈+|草|艹|笑死|乐|确实|对的?|是的|好耶|啊这|1+|\+1|6+|牛|"
    r"绝了|太强了|离谱|难绷|绷)[\s。.!！~～…]*$",
)

_FLAGS = re.IGNORECASE | re.MULTILINE


def matches_any(text: str, patterns: Sequence[str]) -> bool:
    """文本是否命中任意一条正则（非法正则会跳过而不是抛错）。"""
    if not text:
        return False
    for pattern in patterns:
        if not pattern:
            continue
        try:
            if re.search(pattern, text, flags=_FLAGS):
                return True
        except re.error:
            continue
    return False


@dataclass(frozen=True, slots=True)
class Signals:
    """机器人那条消息之后，现场给出的反应信号。"""

    has_reply: bool = False
    confusion: bool = False
    filler_only: bool = False
    first_after: ChatLine | None = None


def analyze_reaction(
    lines: Sequence[ChatLine],
    utterance_end: int,
    confusion_patterns: Sequence[str] = DEFAULT_CONFUSION_PATTERNS,
    filler_patterns: Sequence[str] = DEFAULT_FILLER_PATTERNS,
    lookahead: int = 3,
) -> Signals:
    """看看机器人说完之后，别人是怎么接的。"""
    after = [line for line in lines[utterance_end + 1 :] if not line.is_self]
    if not after:
        return Signals()
    window = after[: max(1, lookahead)]
    confusion = any(matches_any(line.text, confusion_patterns) for line in window)
    filler_only = matches_any(after[0].text, filler_patterns) and not confusion
    return Signals(
        has_reply=True,
        confusion=confusion,
        filler_only=filler_only,
        first_after=after[0],
    )


def format_transcript(lines: Sequence[ChatLine], max_chars: int = 220) -> str:
    """把聊天记录排成带行号的文本。"""
    rows = []
    for index, line in enumerate(lines, 1):
        text = line.text
        if max_chars and len(text) > max_chars:
            text = text[: max(1, max_chars - 1)] + "…"
        rows.append(f"#{index} {line.speaker()}：{text}")
    return "\n".join(rows)


def build_user_prompt(
    template: str,
    session_label: str,
    transcript: str,
    target_indexes: Sequence[int],
    reaction: str = "",
) -> str:
    """填充用户提示词模板。"""
    target = "、".join(str(index) for index in target_indexes) or "（未知）"
    return (
        template.replace("{{session_label}}", session_label)
        .replace("{{transcript}}", transcript)
        .replace("{{target}}", target)
        .replace("{{reaction}}", reaction or "（机器人说完之后，暂时没有人接话）")
    )


# ---------------------------------------------------------------------------
# 解析 LLM 输出
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Verdict:
    recall: bool
    confidence: float
    reason: str
    raw: str = ""


_RECALL_WORDS = {"recall", "撤回", "true", "yes", "y", "1", "should_recall"}
_KEEP_WORDS = {"keep", "保留", "false", "no", "n", "0", "should_keep"}


def _iter_json_objects(text: str) -> list[str]:
    """从文本里由后往前抠出可能的 JSON 对象片段。"""
    return [match.group(0) for match in re.finditer(r"\{[\s\S]{0,800}?\}", text)][::-1]


def parse_verdict(raw: str) -> Verdict:
    """把模型输出解析成裁决。解析失败一律按 keep 处理（宁可不撤）。"""
    text = (raw or "").strip()
    if not text:
        return Verdict(False, 0.0, "空响应", raw)
    for blob in _iter_json_objects(text):
        try:
            data = json.loads(blob)
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        for key in ("verdict", "decision", "result", "action"):
            value = data.get(key)
            if isinstance(value, bool):
                verdict_raw = "recall" if value else "keep"
            elif isinstance(value, str):
                verdict_raw = value.strip().lower()
            else:
                continue
            confidence = _coerce_confidence(data.get("confidence"))
            reason = str(data.get("reason") or data.get("why") or "").strip()
            if verdict_raw in _RECALL_WORDS:
                return Verdict(True, confidence, truncate_reason(reason), raw)
            if verdict_raw in _KEEP_WORDS:
                return Verdict(False, confidence, truncate_reason(reason), raw)
    lowered = text.lower()
    keep_hint = "keep" in lowered or "保留" in text or "不该撤" in text
    recall_hint = "recall" in lowered or "撤回" in text or "该撤" in text
    if recall_hint and not keep_hint:
        return Verdict(True, 0.6, truncate_reason(text), raw)
    return Verdict(False, 0.0, truncate_reason(text), raw)


def _coerce_confidence(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
    if number > 1.0:
        number = number / 100.0 if number <= 100.0 else 1.0
    return max(0.0, min(1.0, number))


def truncate_reason(reason: str, limit: int = 60) -> str:
    reason = " ".join((reason or "").split())
    if len(reason) > limit:
        reason = reason[: limit - 1] + "…"
    return reason
