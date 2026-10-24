"""插件配置的读取与归一化。

AstrBot 的插件配置是嵌套 dict，这里统一摊平成带默认值的 Settings，
避免在处理逻辑里到处写 ``.get(...) or 默认值``。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .judge import (
    DEFAULT_CONFUSION_PATTERNS,
    DEFAULT_FILLER_PATTERNS,
    DEFAULT_INSTANT_PATTERNS,
    DEFAULT_REQUEST_PATTERNS,
    DEFAULT_USER_PROMPT,
)
from .precheck import (
    DEFAULT_JUDGE_PROMPT,
    DEFAULT_REWRITE_PROMPT,
    DEFAULT_TOOL_PATTERNS,
)

_MISSING = object()


def dig(data: object, path: str, default: object = None) -> object:
    """按 ``a.b.c`` 取嵌套值，缺失时返回默认值。"""
    current: object = data
    for key in path.split("."):
        if not isinstance(current, Mapping):
            return default
        value = current.get(key, _MISSING)
        if value is _MISSING:
            return default
        current = value
    return current


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on", "是", "开"}:
            return True
        if text in {"0", "false", "no", "off", "否", "关"}:
            return False
    return default


def _as_int(value: object, default: int, minimum: int | None = None) -> int:
    try:
        number = int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if minimum is not None and number < minimum:
        return minimum
    return number


def _as_float(value: object, default: float, minimum: float | None = None) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if minimum is not None and number < minimum:
        return minimum
    return number


def _as_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_patterns(value: object, default: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [part.strip() for part in value.splitlines()]
    elif isinstance(value, (list, tuple)):
        items = [_as_str(item) for item in value]
    else:
        return tuple(default)
    items = [item for item in items if item]
    return tuple(items) if items else tuple(default)


def _as_list(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [part.strip() for part in value.replace("，", ",").split(",")]
    elif isinstance(value, (list, tuple)):
        items = [_as_str(item) for item in value]
    else:
        return ()
    return tuple(item for item in items if item)


@dataclass(frozen=True, slots=True)
class PrecheckSettings:
    """发送前逻辑审查的参数。"""

    enable: bool = True
    dry_run: bool = False
    max_attempts: int = 1
    history_count: int = 10
    max_line_chars: int = 120
    min_chars: int = 2
    timeout: float = 12.0
    max_checks_per_hour: int = 200
    skip_filler: bool = True
    skip_structured: bool = True
    tool_patterns: tuple[str, ...] = DEFAULT_TOOL_PATTERNS
    judge_prompt: str = DEFAULT_JUDGE_PROMPT
    rewrite_prompt: str = DEFAULT_REWRITE_PROMPT

    @classmethod
    def from_config(cls, data: Mapping) -> PrecheckSettings:
        return cls(
            enable=_as_bool(dig(data, "precheck.enable"), True),
            dry_run=_as_bool(dig(data, "precheck.dry_run"), False),
            max_attempts=_as_int(dig(data, "precheck.max_attempts"), 1, 0),
            history_count=_as_int(dig(data, "precheck.history_count"), 10, 2),
            max_line_chars=_as_int(dig(data, "precheck.max_line_chars"), 120, 20),
            min_chars=_as_int(dig(data, "precheck.min_chars"), 2, 1),
            timeout=_as_float(dig(data, "precheck.timeout_seconds"), 12.0, 3.0),
            max_checks_per_hour=_as_int(
                dig(data, "precheck.max_checks_per_hour"), 200, 1
            ),
            skip_filler=_as_bool(dig(data, "precheck.skip_filler"), True),
            skip_structured=_as_bool(dig(data, "precheck.skip_structured"), True),
            tool_patterns=_as_patterns(
                dig(data, "precheck.skip_tool_patterns"), DEFAULT_TOOL_PATTERNS
            ),
            judge_prompt=_as_str(dig(data, "precheck.judge_prompt"))
            or DEFAULT_JUDGE_PROMPT,
            rewrite_prompt=_as_str(dig(data, "precheck.rewrite_prompt"))
            or DEFAULT_REWRITE_PROMPT,
        )


@dataclass(frozen=True, slots=True)
class Settings:
    """插件运行参数。"""

    enabled: bool = True
    dry_run: bool = False
    recall_groups: bool = True
    recall_private: bool = True
    only_sessions: tuple[str, ...] = ()
    ignore_sessions: tuple[str, ...] = ()

    max_message_age: float = 100.0
    min_message_age: float = 2.0
    reaction_window: float = 75.0
    history_count: int = 25
    max_line_chars: int = 220

    min_confidence: float = 0.78
    llm_provider_id: str = ""
    llm_timeout: float = 30.0
    judge_cooldown: float = 6.0
    max_judgments_per_hour: int = 30
    rejudge_on_confusion: bool = True

    max_recalls_per_hour: int = 3
    max_recalls_per_day: int = 12
    session_cooldown: float = 300.0
    max_batch_recall: int = 4

    instant_patterns: tuple[str, ...] = DEFAULT_INSTANT_PATTERNS
    confusion_patterns: tuple[str, ...] = DEFAULT_CONFUSION_PATTERNS
    request_patterns: tuple[str, ...] = DEFAULT_REQUEST_PATTERNS
    filler_patterns: tuple[str, ...] = DEFAULT_FILLER_PATTERNS
    skip_filler_reply: bool = True
    honor_explicit_request: bool = True

    recall_notice: str = ""
    log_decisions: bool = True
    user_prompt: str = DEFAULT_USER_PROMPT
    precheck: PrecheckSettings = field(default_factory=PrecheckSettings)
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Mapping | None) -> Settings:
        data: Mapping = config if isinstance(config, Mapping) else {}
        return cls(
            enabled=_as_bool(dig(data, "enabled"), True),
            dry_run=_as_bool(dig(data, "dry_run"), False),
            recall_groups=_as_bool(dig(data, "scope.recall_groups"), True),
            recall_private=_as_bool(dig(data, "scope.recall_private"), True),
            only_sessions=_as_list(dig(data, "scope.only_sessions")),
            ignore_sessions=_as_list(dig(data, "scope.ignore_sessions")),
            max_message_age=_as_float(
                dig(data, "timing.max_message_age_seconds"), 100.0, 5.0
            ),
            min_message_age=_as_float(
                dig(data, "timing.min_message_age_seconds"), 2.0, 0.0
            ),
            reaction_window=_as_float(
                dig(data, "timing.reaction_window_seconds"), 75.0, 5.0
            ),
            history_count=_as_int(dig(data, "timing.history_count"), 25, 4),
            max_line_chars=_as_int(dig(data, "timing.max_line_chars"), 220, 20),
            min_confidence=_as_float(dig(data, "decision.min_confidence"), 0.78, 0.0),
            llm_provider_id=_as_str(dig(data, "decision.llm_provider_id")),
            llm_timeout=_as_float(dig(data, "decision.llm_timeout_seconds"), 30.0, 5.0),
            judge_cooldown=_as_float(
                dig(data, "decision.judge_cooldown_seconds"), 6.0, 0.0
            ),
            max_judgments_per_hour=_as_int(
                dig(data, "decision.max_judgments_per_hour"), 30, 1
            ),
            rejudge_on_confusion=_as_bool(
                dig(data, "decision.rejudge_on_confusion"), True
            ),
            max_recalls_per_hour=_as_int(
                dig(data, "limits.max_recalls_per_hour"), 3, 0
            ),
            max_recalls_per_day=_as_int(dig(data, "limits.max_recalls_per_day"), 12, 0),
            session_cooldown=_as_float(
                dig(data, "limits.session_cooldown_seconds"), 300.0, 0.0
            ),
            max_batch_recall=_as_int(dig(data, "limits.max_batch_recall"), 4, 1),
            instant_patterns=_as_patterns(
                dig(data, "signals.instant_recall_patterns"), DEFAULT_INSTANT_PATTERNS
            ),
            confusion_patterns=_as_patterns(
                dig(data, "signals.confusion_patterns"), DEFAULT_CONFUSION_PATTERNS
            ),
            request_patterns=_as_patterns(
                dig(data, "signals.explicit_request_patterns"), DEFAULT_REQUEST_PATTERNS
            ),
            filler_patterns=_as_patterns(
                dig(data, "signals.filler_patterns"), DEFAULT_FILLER_PATTERNS
            ),
            skip_filler_reply=_as_bool(dig(data, "signals.skip_filler_reply"), True),
            honor_explicit_request=_as_bool(
                dig(data, "signals.honor_explicit_request"), True
            ),
            recall_notice=_as_str(dig(data, "output.recall_notice")),
            log_decisions=_as_bool(dig(data, "output.log_decisions"), True),
            user_prompt=_as_str(dig(data, "output.judge_prompt"))
            or DEFAULT_USER_PROMPT,
            precheck=PrecheckSettings.from_config(data),
        )

    # -- 会话过滤 ---------------------------------------------------------
    def session_allowed(self, *, umo: str, peer_id: str, is_group: bool) -> bool:
        if is_group and not self.recall_groups:
            return False
        if not is_group and not self.recall_private:
            return False
        if self.only_sessions and not any(
            _session_matches(pattern, umo, peer_id) for pattern in self.only_sessions
        ):
            return False
        return not any(
            _session_matches(pattern, umo, peer_id) for pattern in self.ignore_sessions
        )


def _session_matches(pattern: str, umo: str, peer_id: str) -> bool:
    pattern = (pattern or "").strip()
    if not pattern:
        return False
    if pattern in {umo, peer_id}:
        return True
    if umo.endswith(":" + pattern):
        return True
    return bool(pattern.endswith(":") and umo.startswith(pattern))
