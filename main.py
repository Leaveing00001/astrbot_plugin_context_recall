"""读空气：发送前逻辑审查 + 发送后语境撤回。

两条链路：
  1. 发送前（precheck）：截下即将发出的消息 → LLM 判是否明显不合逻辑
     → 不合格就打回重写 → 重写的再判一遍 → 始终不过就拦截不发送
  2. 发送后（recall）：有人接话时回看自己上一条，若明显与语境不符则撤回

判定统一走 OneBot 的聊天流水（get_*_msg_history），不依赖 AstrBot 内部上下文。
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star, StarTools, register

from .core.guard import AuditLog, RecallBudget
from .core.history import HistoryUnavailable, fetch_history
from .core.judge import (
    SYSTEM_PROMPT,
    analyze_reaction,
    build_user_prompt,
    format_transcript,
    matches_any,
    parse_verdict,
    truncate_reason,
)
from .core.models import ChatLine, normalize_messages
from .core.precheck import (
    JUDGE_SYSTEM,
    REWRITE_SYSTEM,
    build_judge_prompt,
    build_rewrite_prompt,
    clean_rewrite,
    has_tool_markers,
    parse_verdict_line,
)
from .core.settings import PrecheckSettings, Settings

PLUGIN_NAME = "astrbot_plugin_context_recall"
AUTHOR = "DeepSeek Harness"
DESC = "让 bot 读空气：发送前审查逻辑、发送后按语境撤回"
VERSION = "1.1.0"

TAG = "[读空气]"

#: 连续多条发送（分段回复）之间允许的最大间隔，超过就当成另一段话
_UTTERANCE_GAP = 10.0

#: 拉取聊天流水的超时
_HISTORY_TIMEOUT = 8.0

#: 撤回动作的超时
_RECALL_TIMEOUT = 8.0

#: 后台任务并发上限，防止消息高峰打爆
_MAX_BACKGROUND_TASKS = 6


@dataclass(slots=True)
class Trigger:
    """触发一次复核的那条消息。"""

    umo: str
    platform_id: str
    self_id: str
    sender_id: str
    sender_name: str
    is_group: bool
    peer_id: str
    session_label: str
    message_id: str
    text: str
    ts: float
    received_at: float
    bot: Any

    def as_line(self) -> ChatLine:
        return ChatLine(
            message_id=self.message_id,
            user_id=self.sender_id,
            name=self.sender_name,
            ts=self.ts,
            text=self.text,
            is_self=False,
        )


@dataclass(slots=True)
class ReviewResult:
    """一次复核的结论（也用于 /air test 的输出）。"""

    stage: str
    verdict: str = ""
    confidence: float = 0.0
    reason: str = ""
    action: str = ""
    transcript: str = ""
    target_indexes: list[int] = field(default_factory=list)
    session_label: str = ""

    @property
    def ok(self) -> bool:
        return self.stage == "done"


@register(PLUGIN_NAME, AUTHOR, DESC, VERSION)
class ContextRecallPlugin(Star):
    """读空气撤回插件。"""

    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.cfg = Settings.from_config(config)
        self._data_dir = self._resolve_data_dir()
        self._state_path = (
            self._data_dir / "state.json" if self._data_dir is not None else None
        )
        audit_path = (
            self._data_dir / "audit.jsonl" if self._data_dir is not None else None
        )
        self._budget = RecallBudget(
            per_hour=self.cfg.max_recalls_per_hour,
            per_day=self.cfg.max_recalls_per_day,
            session_cooldown=self.cfg.session_cooldown,
            judgments_per_hour=self.cfg.max_judgments_per_hour,
        )
        self._audit = AuditLog(audit_path)
        self._seen: dict[str, dict[str, Any]] = {}
        self._tasks: set[asyncio.Task] = set()
        self._history_down_until = 0.0
        self._warned: set[str] = set()
        self._session_judged_at: dict[str, float] = {}
        self._precheck_marks: deque[float] = deque(maxlen=2048)
        self._precheck_stats: dict[str, int] = {
            "checked": 0,
            "passed": 0,
            "rewritten": 0,
            "blocked": 0,
        }
        self._load_state()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        cfg = self.cfg
        self.logger.info(
            "%s 已加载 | 启用=%s 演练=%s 群聊=%s 私聊=%s 窗口=%.0fs 置信度=%.2f "
            "配额=%d/h·%d/d 流水上限=%d 条",
            TAG,
            cfg.enabled,
            cfg.dry_run,
            cfg.recall_groups,
            cfg.recall_private,
            cfg.max_message_age,
            cfg.min_confidence,
            cfg.max_recalls_per_hour,
            cfg.max_recalls_per_day,
            cfg.history_count,
        )
        if not cfg.enabled:
            self.logger.info("%s 配置里 enabled=false，插件将不做任何事。", TAG)
        pc = cfg.precheck
        self.logger.info(
            "%s 发送前审查：%s | 演练=%s 重写上限=%d 次 超时=%.0fs 上下文=%d 条",
            TAG,
            "开" if pc.enable else "关",
            pc.dry_run,
            pc.max_attempts,
            pc.timeout,
            pc.history_count,
        )

    async def terminate(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        self._save_state()
        self.logger.info("%s 已卸载，本次共撤回 %d 次。", TAG, self._budget.recalls_last_hour(time.time()))

    # ------------------------------------------------------------------
    # 主入口：任何消息都先看一眼（发送后复核）
    # ------------------------------------------------------------------
    @filter.event_message_type(EventMessageType.ALL)
    async def on_any_message(self, event: AstrMessageEvent):
        if not self.cfg.enabled:
            return
        trigger = self._collect(event)
        if trigger is None:
            return
        self._spawn(self._review(trigger))

    # ------------------------------------------------------------------
    # 发送前：把即将发出的消息截下来审一遍
    # ------------------------------------------------------------------
    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        if not (self.cfg.enabled and self.cfg.precheck.enable):
            return
        try:
            await self._precheck(event)
        except Exception:  # noqa: BLE001 - 审查绝不能把正常回复搞挂
            self.logger.exception("%s 发送前审查异常，本条按原样放行", TAG)

    # ------------------------------------------------------------------
    # 采集
    # ------------------------------------------------------------------
    def _collect(self, event: AstrMessageEvent) -> Trigger | None:
        try:
            self_id = str(event.get_self_id() or "")
            sender_id = str(event.get_sender_id() or "")
            group_id = str(event.get_group_id() or "")
        except Exception:  # noqa: BLE001 - 事件字段不可控
            return None
        if not self_id or not sender_id or sender_id == self_id:
            return None

        is_group = bool(group_id)
        peer_id = group_id if is_group else sender_id
        if not peer_id:
            return None

        umo = str(getattr(event, "unified_msg_origin", "") or "")
        if not umo or not self.cfg.session_allowed(
            umo=umo, peer_id=peer_id, is_group=is_group
        ):
            return None

        platform_id = ""
        with contextlib.suppress(Exception):
            platform_id = str(event.get_platform_id() or "")
        bot = self._resolve_bot(event, platform_id)
        if bot is None:
            return None

        message_obj = getattr(event, "message_obj", None)
        message_id = str(getattr(message_obj, "message_id", "") or "")
        text = ""
        with contextlib.suppress(Exception):
            text = str(event.get_message_str() or "")
        ts = 0.0
        with contextlib.suppress(TypeError, ValueError):
            ts = float(getattr(message_obj, "timestamp", 0) or 0)
        received_at = time.time()
        if not ts or ts > received_at + 300:
            ts = received_at

        sender_name = ""
        with contextlib.suppress(Exception):
            sender_name = str(event.get_sender_name() or "")

        return Trigger(
            umo=umo,
            platform_id=platform_id,
            self_id=self_id,
            sender_id=sender_id,
            sender_name=sender_name or sender_id,
            is_group=is_group,
            peer_id=peer_id,
            session_label=self._session_label(event, is_group, peer_id),
            message_id=message_id,
            text=text,
            ts=ts,
            received_at=received_at,
            bot=bot,
        )

    def _resolve_bot(self, event: AstrMessageEvent, platform_id: str) -> Any:
        bot = getattr(event, "bot", None)
        if bot is not None and hasattr(bot, "call_action"):
            return bot
        if not platform_id:
            return None
        with contextlib.suppress(Exception):
            platform = self.context.get_platform_inst(platform_id)
            if platform is not None and hasattr(platform, "get_client"):
                return platform.get_client()
        return None

    @staticmethod
    def _session_label(event: AstrMessageEvent, is_group: bool, peer_id: str) -> str:
        if not is_group:
            return f"私聊 {peer_id}"
        name = ""
        with contextlib.suppress(Exception):
            group = getattr(getattr(event, "message_obj", None), "group", None)
            name = str(getattr(group, "group_name", "") or "")
        return f"群聊「{name}」" if name else f"群聊 {peer_id}"

    def _spawn(self, coro) -> None:
        if len(self._tasks) >= _MAX_BACKGROUND_TASKS:
            return
        task = asyncio.create_task(self._guard(coro))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @staticmethod
    async def _guard(coro) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - 后台任务绝不能把异常抛给 AstrBot
            logger.exception("%s 后台复核任务异常", TAG)

    # ------------------------------------------------------------------
    # 发送前审查
    # ------------------------------------------------------------------
    #: 解除 skip_structured 之后仍然允许一起发出去的结构化组件
    _TRANSPARENT_COMPONENTS = frozenset({"At", "Reply"})

    async def _precheck(self, event: AstrMessageEvent) -> None:
        """过审才放行：判 → 不合格则重写 → 重写的再判 → 始终不过就拦下不发。"""
        pc = self.cfg.precheck
        result = event.get_result()
        if result is None or not result.chain:
            return
        if not result.is_llm_result():
            return  # 指令回显 / 插件结果 / 流式结果，一律不碰

        # 结构化消息（引用回复 / @某人）是工具调用的产物，整条放行：
        # 既不判、也不改写、更不拦截，免得把 reply_to / [At:123] 改坏。
        for comp in result.chain:
            name = type(comp).__name__
            if name == "Plain":
                continue
            if not pc.skip_structured and name in self._TRANSPARENT_COMPONENTS:
                continue
            return

        plains = [comp for comp in result.chain if isinstance(comp, Plain)]
        if not plains:
            return
        draft = "".join(comp.text or "" for comp in plains).strip()
        if len(draft) < pc.min_chars:
            return
        if has_tool_markers(draft, pc.tool_patterns):
            self.logger.debug("%s 含工具调用痕迹，直接放行", TAG)
            return
        if pc.skip_filler and matches_any(draft, self.cfg.filler_patterns):
            return

        trigger = self._collect(event)
        if trigger is None:
            return
        if not self._precheck_budget_ok():
            self._warn_once(
                "precheck_budget", "发送前审查已达本小时上限，后续消息直接放行"
            )
            return

        provider_id = await self._resolve_provider(trigger.umo)
        if not provider_id:
            return
        raw = await self._fetch(trigger, count=pc.history_count)
        if raw is None:
            return
        lines = normalize_messages(raw, trigger.self_id, pc.max_line_chars)
        transcript = format_transcript(lines, pc.max_line_chars) if lines else "（暂无）"

        current = draft
        attempts = 0
        reason = ""
        while True:
            self._precheck_stats["checked"] += 1
            self._precheck_marks.append(time.time())
            verdict = await self._judge_reply(
                provider_id, trigger, transcript, current, pc
            )
            if verdict is None:
                return  # 调用异常：放行，绝不因为审查失败而吞消息
            if verdict.ok:
                break
            reason = verdict.reason
            if attempts >= pc.max_attempts:
                self._block_reply(event, trigger, current, reason, attempts)
                return
            attempts += 1
            rewritten = await self._rewrite_reply(
                provider_id, trigger, transcript, current, reason, pc
            )
            if not rewritten:
                self._block_reply(event, trigger, current, reason or "重写失败", attempts)
                return
            current = rewritten

        if current != draft:
            self._apply_text(result, plains, current)
            self._precheck_stats["rewritten"] += 1
            self.logger.info(
                "%s 重写后放行 | %s | 第 %d 次 | 原：%s → 新：%s",
                TAG,
                trigger.session_label,
                attempts,
                truncate_reason(draft, 30),
                truncate_reason(current, 30),
            )
            self._audit_precheck(trigger, draft, current, "rewritten", reason, attempts)
            return
        self._precheck_stats["passed"] += 1

    async def _judge_reply(
        self,
        provider_id: str,
        trigger: Trigger,
        transcript: str,
        draft: str,
        pc: PrecheckSettings,
    ):
        prompt = build_judge_prompt(
            pc.judge_prompt, trigger.session_label, transcript, draft
        )
        try:
            response = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                    system_prompt=JUDGE_SYSTEM,
                ),
                timeout=pc.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            self._warn_once("precheck_llm", f"发送前审查调用失败，本次放行：{exc}")
            return None
        return parse_verdict_line(str(getattr(response, "completion_text", "") or ""))

    async def _rewrite_reply(
        self,
        provider_id: str,
        trigger: Trigger,
        transcript: str,
        draft: str,
        reason: str,
        pc: PrecheckSettings,
    ) -> str:
        prompt = build_rewrite_prompt(
            pc.rewrite_prompt, trigger.session_label, transcript, draft, reason
        )
        try:
            response = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                    system_prompt=REWRITE_SYSTEM,
                ),
                timeout=pc.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            self._warn_once("precheck_rewrite", f"重写调用失败：{exc}")
            return ""
        return clean_rewrite(str(getattr(response, "completion_text", "") or ""))

    def _block_reply(
        self,
        event: AstrMessageEvent,
        trigger: Trigger,
        draft: str,
        reason: str,
        attempts: int,
    ) -> None:
        pc = self.cfg.precheck
        self._precheck_stats["blocked"] += 1
        if pc.dry_run:
            self.logger.info(
                "%s 演练：本该拦截 | %s | 理由：%s | 草稿：%s",
                TAG,
                trigger.session_label,
                reason,
                truncate_reason(draft, 60),
            )
            self._audit_precheck(trigger, draft, "", "dry_run", reason, attempts)
            return
        event.clear_result()
        self.logger.info(
            "%s 已拦截（不发送）| %s | 理由：%s | 草稿：%s",
            TAG,
            trigger.session_label,
            reason,
            truncate_reason(draft, 60),
        )
        self._audit_precheck(trigger, draft, "", "blocked", reason, attempts)

    @staticmethod
    def _apply_text(
        result: Any, plains: list[Plain], text: str
    ) -> None:
        first = plains[0]
        first.text = text
        keep = id(first)
        result.chain[:] = [
            comp
            for comp in result.chain
            if not isinstance(comp, Plain) or id(comp) == keep
        ]

    def _precheck_budget_ok(self) -> bool:
        now = time.time()
        marks = self._precheck_marks
        while marks and now - marks[0] > 3600:
            marks.popleft()
        return len(marks) < self.cfg.precheck.max_checks_per_hour

    def _audit_precheck(
        self,
        trigger: Trigger,
        draft: str,
        final: str,
        action: str,
        reason: str,
        attempts: int,
    ) -> None:
        self._audit.add(
            {
                "phase": "precheck",
                "umo": trigger.umo,
                "session": trigger.session_label,
                "draft": truncate_reason(draft, 160),
                "final": truncate_reason(final, 160),
                "verdict": "abnormal" if action in {"blocked", "dry_run"} else "ok",
                "action": action,
                "attempts": attempts,
                "reason": truncate_reason(reason, 120),
            }
        )

    # ------------------------------------------------------------------
    # 复核
    # ------------------------------------------------------------------
    async def _review(
        self,
        trigger: Trigger,
        *,
        dry_run_override: bool | None = None,
        force: bool = False,
    ) -> ReviewResult:
        cfg = self.cfg
        track = dry_run_override is None and not force
        now = time.time()
        if now - trigger.received_at > cfg.max_message_age:
            return ReviewResult(stage="stale")

        raw = await self._fetch(trigger)
        if raw is None:
            return ReviewResult(stage="history_unavailable")

        lines = normalize_messages(raw, trigger.self_id, cfg.max_line_chars)
        if trigger.message_id and not any(
            line.message_id == trigger.message_id for line in lines
        ):
            lines.append(trigger.as_line())
            lines.sort(key=lambda line: line.ts)
        if not lines:
            return ReviewResult(stage="no_history")

        utterance = self._pick_utterance(lines, now)
        if not utterance:
            return ReviewResult(stage="no_recent_self_message")

        end_index = self._index_of(lines, utterance[-1])
        signals = analyze_reaction(
            lines,
            end_index,
            confusion_patterns=cfg.confusion_patterns,
            filler_patterns=cfg.filler_patterns,
        )
        if not signals.has_reply:
            return ReviewResult(stage="no_reaction")

        if trigger.received_at - utterance[-1].ts > cfg.reaction_window:
            return ReviewResult(stage="reaction_too_late")

        key = self._utterance_key(trigger.umo, utterance)
        if track:
            entry = self._seen.get(key)
            if entry:
                if entry.get("done"):
                    return ReviewResult(stage="already_handled")
                if trigger.message_id and trigger.message_id in entry["triggers"]:
                    return ReviewResult(stage="duplicate_trigger")
                judges = int(entry.get("judges", 0))
                if judges >= 2:
                    return ReviewResult(stage="judged_twice")
                if judges >= 1 and not (
                    cfg.rejudge_on_confusion and signals.confusion
                ):
                    return ReviewResult(stage="already_judged")

        our_text = "\n".join(line.text for line in utterance)
        transcript = format_transcript(lines, cfg.max_line_chars)
        target_indexes = self._target_indexes(lines, utterance)
        base = ReviewResult(
            stage="done",
            transcript=transcript,
            target_indexes=target_indexes,
            session_label=trigger.session_label,
        )

        def skip(note: str) -> ReviewResult:
            return self._finish(base, verdict="", reason=note)

        # 一档：机器人自己吐出了模板痕迹，无需模型判断
        if matches_any(our_text, cfg.instant_patterns):
            return await self._recall(
                trigger,
                utterance,
                key=key,
                reason="输出里出现了模板/提示词痕迹",
                confidence=1.0,
                source="instant",
                dry_run_override=dry_run_override,
                base=base,
                track=track,
            )

        # 二档：有人明确要它撤回
        if cfg.honor_explicit_request and matches_any(trigger.text, cfg.request_patterns):
            return await self._recall(
                trigger,
                utterance,
                key=key,
                reason="用户明确要求撤回",
                confidence=0.99,
                source="request",
                dry_run_override=dry_run_override,
                base=base,
                track=track,
            )

        # 敷衍式接话说明对话正常继续，不必花钱判断
        if cfg.skip_filler_reply and signals.filler_only:
            if track:
                self._note(key, trigger_id=trigger.message_id)
            return skip("对方只是随口接话")

        if not self._budget.allow_judgement(now):
            if track:
                self._note(key, trigger_id=trigger.message_id)
            return skip("本小时判定次数已达上限")

        last_judged = self._session_judged_at.get(trigger.umo, 0.0)
        if cfg.judge_cooldown and now - last_judged < cfg.judge_cooldown:
            if track:
                self._note(key, trigger_id=trigger.message_id)
            return skip("同会话判定冷却中")

        provider_id = await self._resolve_provider(trigger.umo)
        if not provider_id:
            self._warn_once("provider", "找不到可用的 LLM Provider，无法读空气")
            return skip("没有可用的 LLM Provider")

        prompt = build_user_prompt(
            cfg.user_prompt,
            trigger.session_label,
            transcript,
            target_indexes,
            self._reaction_text(signals),
        )
        self._budget.note_judgement(now)
        self._session_judged_at[trigger.umo] = now
        verdict = await self._ask_llm(provider_id, prompt)
        if verdict is None:
            if track:
                self._note(key, trigger_id=trigger.message_id)
            return skip("模型调用失败")

        base.verdict = "recall" if verdict.recall else "keep"
        base.confidence = verdict.confidence
        base.reason = verdict.reason

        if verdict.recall and verdict.confidence >= cfg.min_confidence:
            return await self._recall(
                trigger,
                utterance,
                key=key,
                reason=verdict.reason or "与当下语境不符",
                confidence=verdict.confidence,
                source="llm",
                dry_run_override=dry_run_override,
                base=base,
                track=track,
            )

        if track:
            self._note(key, trigger_id=trigger.message_id, judges=1)
            self._audit_record(
                trigger,
                utterance,
                verdict="keep",
                action="kept",
                reason=verdict.reason,
                confidence=verdict.confidence,
                source="llm",
            )
        if cfg.log_decisions:
            self.logger.info(
                "%s 保留 | %s | 置信度 %.2f | %s",
                TAG,
                trigger.session_label,
                verdict.confidence,
                verdict.reason or "无明显问题",
            )
        base.action = "kept"
        return base

    def _pick_utterance(self, lines: list[ChatLine], now: float) -> list[ChatLine]:
        cfg = self.cfg
        recent = [
            line
            for line in lines
            if line.is_self and 0 <= now - line.ts <= cfg.max_message_age
        ]
        if not recent:
            return []
        recent.sort(key=lambda line: line.ts)
        group = [recent[-1]]
        for line in reversed(recent[:-1]):
            if group[0].ts - line.ts <= _UTTERANCE_GAP:
                group.insert(0, line)
            else:
                break
        group = group[-cfg.max_batch_recall :]
        if now - group[0].ts < cfg.min_message_age:
            return []
        return group

    @staticmethod
    def _index_of(lines: list[ChatLine], target: ChatLine) -> int:
        for index, line in enumerate(lines):
            if line is target:
                return index
        return len(lines) - 1

    @staticmethod
    def _target_indexes(lines: list[ChatLine], utterance: list[ChatLine]) -> list[int]:
        ids = {id(line) for line in utterance}
        return [index for index, line in enumerate(lines, 1) if id(line) in ids]

    @staticmethod
    def _reaction_text(signals) -> str:
        """把「现场反应」翻译成一句人能看懂的话，交给模型当线索。"""
        first = signals.first_after
        if first is None:
            return ""
        who = first.speaker()
        snippet = truncate_reason(first.text or "（非文字消息）", 30)
        if signals.confusion:
            return f"{who} 随后回了一句「{snippet}」，看起来是没听懂、或者觉得答错了。"
        if signals.filler_only:
            return f"{who} 随口应了一句「{snippet}」，对话在正常继续。"
        return f"{who} 接着说了「{snippet}」。"

    @staticmethod
    def _utterance_key(umo: str, utterance: list[ChatLine]) -> str:
        return umo + "#" + ",".join(line.message_id for line in utterance)

    # ------------------------------------------------------------------
    # 拉取聊天流水
    # ------------------------------------------------------------------
    async def _fetch(self, trigger: Trigger, count: int | None = None) -> list[dict] | None:
        now = time.time()
        if now < self._history_down_until:
            return None
        try:
            raw = await asyncio.wait_for(
                fetch_history(
                    trigger.bot,
                    is_group=trigger.is_group,
                    peer_id=trigger.peer_id,
                    count=count or self.cfg.history_count,
                ),
                timeout=_HISTORY_TIMEOUT,
            )
        except (HistoryUnavailable, asyncio.TimeoutError) as exc:
            self._note_history_failure(exc)
            return None
        except Exception as exc:  # noqa: BLE001
            self._note_history_failure(exc)
            return None
        return raw

    def _note_history_failure(self, exc: object) -> None:
        self._history_down_until = time.time() + 900
        self._warn_once(
            "history",
            f"读取聊天流水失败，本插件需要 OneBot 的 get_group_msg_history / "
            f"get_friend_msg_history 接口；15 分钟内不再重试。原因：{exc}",
        )

    # ------------------------------------------------------------------
    # 模型
    # ------------------------------------------------------------------
    async def _resolve_provider(self, umo: str) -> str | None:
        if self.cfg.llm_provider_id:
            return self.cfg.llm_provider_id
        getter = getattr(self.context, "get_current_chat_provider_id", None)
        if callable(getter):
            with contextlib.suppress(Exception):
                provider_id = getter(umo)
                if inspect.isawaitable(provider_id):
                    provider_id = await provider_id
                if provider_id:
                    return str(provider_id)
        with contextlib.suppress(Exception):
            provider = self.context.get_using_provider(umo)
            if provider is not None:
                return str(provider.meta().id)
        return None

    async def _ask_llm(self, provider_id: str, prompt: str):
        try:
            response = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                ),
                timeout=self.cfg.llm_timeout,
            )
        except asyncio.TimeoutError:
            self._warn_once("llm_timeout", "模型判定超时，已跳过本次判定")
            return None
        except Exception as exc:  # noqa: BLE001
            self._warn_once("llm_error", f"模型调用失败：{exc}")
            return None
        text = str(getattr(response, "completion_text", "") or "")
        return parse_verdict(text)

    # ------------------------------------------------------------------
    # 撤回
    # ------------------------------------------------------------------
    async def _recall(
        self,
        trigger: Trigger,
        utterance: list[ChatLine],
        *,
        key: str,
        reason: str,
        confidence: float,
        source: str,
        dry_run_override: bool | None,
        base: ReviewResult,
        track: bool,
    ) -> ReviewResult:
        cfg = self.cfg
        dry_run = cfg.dry_run if dry_run_override is None else dry_run_override
        now = time.time()

        if not dry_run:
            allowed, why = self._budget.allow_recall(trigger.umo, now)
            if not allowed:
                if track:
                    self._note(key, trigger_id=trigger.message_id, done=True)
                    self._audit_record(
                        trigger,
                        utterance,
                        verdict="recall",
                        action="blocked",
                        reason=f"{reason}（{why}）",
                        confidence=confidence,
                        source=source,
                    )
                self.logger.warning("%s 想撤回但被限流：%s", TAG, why)
                base.reason = reason
                base.confidence = confidence
                base.action = "blocked"
                return base

        recalled: list[str] = []
        failed: list[str] = []
        for line in utterance[: cfg.max_batch_recall]:
            message_id = self._as_int(line.message_id)
            if message_id is None:
                failed.append(f"{line.message_id}:非法消息 ID")
                continue
            if dry_run:
                recalled.append(line.message_id)
                continue
            try:
                await asyncio.wait_for(
                    trigger.bot.call_action("delete_msg", message_id=message_id),
                    timeout=_RECALL_TIMEOUT,
                )
                recalled.append(line.message_id)
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{line.message_id}:{exc}")

        if recalled and not dry_run:
            self._budget.note_recall(trigger.umo, now)

        action = "dry_run" if dry_run else ("recalled" if recalled else "failed")

        if track:
            self._note(
                key, trigger_id=trigger.message_id, done=bool(recalled) or dry_run
            )
            self._audit_record(
                trigger,
                utterance,
                verdict="recall",
                action=action,
                reason=f"{reason}｜{failed[0]}" if failed and not recalled else reason,
                confidence=confidence,
                source=source,
            )
        if recalled:
            self.logger.info(
                "%s %s | %s | %d 条 | 置信度 %.2f | %s",
                TAG,
                "演练撤回" if dry_run else "已撤回",
                trigger.session_label,
                len(recalled),
                confidence,
                reason,
            )
        if failed:
            self.logger.warning("%s 撤回失败：%s", TAG, "；".join(failed[:3]))

        if recalled and not dry_run and cfg.recall_notice:
            await self._send_notice(trigger.umo, cfg.recall_notice)

        base.reason = reason
        base.confidence = confidence
        base.action = action
        return base

    async def _send_notice(self, umo: str, text: str) -> None:
        with contextlib.suppress(Exception):
            chain = MessageChain([Plain(text)])
            await self.context.send_message(umo, chain)

    @staticmethod
    def _as_int(value: str) -> int | None:
        text = str(value or "").strip()
        return int(text) if text.isdigit() else None

    # ------------------------------------------------------------------
    # 记账
    # ------------------------------------------------------------------
    def _note(
        self,
        key: str,
        *,
        trigger_id: str = "",
        judges: int = 0,
        done: bool = False,
    ) -> None:
        entry = self._seen.get(key)
        if entry is None:
            entry = {"judges": 0, "done": False, "ts": 0.0, "triggers": set()}
            self._seen[key] = entry
        if trigger_id:
            triggers = entry.get("triggers")
            if not isinstance(triggers, set):
                triggers = set()
                entry["triggers"] = triggers
            triggers.add(trigger_id)
        if judges:
            entry["judges"] = int(entry.get("judges", 0)) + judges
        if done:
            entry["done"] = True
        entry["ts"] = time.time()
        if len(self._seen) > 400:
            self._prune_seen()

    def _prune_seen(self) -> None:
        cutoff = time.time() - 3600
        for key in [k for k, v in self._seen.items() if v.get("ts", 0) < cutoff]:
            self._seen.pop(key, None)

    @staticmethod
    def _finish(base: ReviewResult, *, verdict: str, reason: str) -> ReviewResult:
        base.verdict = verdict
        base.reason = reason
        return base

    def _audit_record(
        self,
        trigger: Trigger,
        utterance: list[ChatLine],
        *,
        verdict: str,
        action: str,
        reason: str,
        confidence: float,
        source: str,
    ) -> None:
        self._audit.add(
            {
                "phase": "recall",
                "umo": trigger.umo,
                "session": trigger.session_label,
                "ids": [line.message_id for line in utterance],
                "text": truncate_reason("\n".join(line.text for line in utterance), 160),
                "trigger": truncate_reason(trigger.text, 120),
                "verdict": verdict,
                "action": action,
                "confidence": round(float(confidence), 3),
                "source": source,
                "reason": truncate_reason(reason, 120),
            }
        )
        if len(self._seen) > 400:
            self._prune_seen()

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        self.logger.warning("%s %s", TAG, message)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def _resolve_data_dir(self):
        with contextlib.suppress(Exception):
            path = StarTools.get_data_dir(PLUGIN_NAME)
            path.mkdir(parents=True, exist_ok=True)
            return path
        return None

    def _load_state(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self._budget.load_state(data)

    def _save_state(self) -> None:
        if self._state_path is None:
            return
        with contextlib.suppress(OSError):
            self._state_path.write_text(
                json.dumps(self._budget.to_state(), ensure_ascii=False),
                encoding="utf-8",
            )

    # ------------------------------------------------------------------
    # 指令
    # ------------------------------------------------------------------
    @filter.command("air", alias={"空气", "读空气"})
    async def air(self, event: AstrMessageEvent):
        """读空气撤回：查看状态、审计流水，或就地试跑一次判定。"""
        sub = self._parse_subcommand(event)
        if sub in {"test", "试跑", "try"}:
            async for result in self._cmd_test(event):
                yield result
            return
        if sub in {"audit", "审计", "log"}:
            async for result in self._cmd_audit(event):
                yield result
            return
        async for result in self._cmd_status(event):
            yield result

    @staticmethod
    def _parse_subcommand(event: AstrMessageEvent) -> str:
        text = ""
        with contextlib.suppress(Exception):
            text = str(event.get_message_str() or "")
        text = re.sub(r"^[\s/!！.。,，]+", "", text).strip()
        for name in ("读空气", "空气", "air"):
            if text.lower().startswith(name.lower()):
                text = text[len(name) :].strip()
                break
        return text.split(" ")[0].lower() if text else ""

    async def _cmd_status(self, event: AstrMessageEvent):
        cfg = self.cfg
        pc = cfg.precheck
        now = time.time()
        used = self._budget.recalls_last_hour(now)
        stats = self._precheck_stats
        lines = [
            f"{TAG} 状态",
            f"启用：{'是' if cfg.enabled else '否'}"
            f" · 撤回演练：{'是（只记日志不真撤）' if cfg.dry_run else '否'}",
            f"范围：群聊{'开' if cfg.recall_groups else '关'} / "
            f"私聊{'开' if cfg.recall_private else '关'}"
            f" · 窗口 {cfg.max_message_age:.0f}s · 置信度阈值 {cfg.min_confidence:.2f}",
            f"发送前审查：{'开' if pc.enable else '关'}"
            f" · 表决行：{stats['checked']} 次，放行 {stats['passed']}"
            f"，重写 {stats['rewritten']}，拦截 {stats['blocked']}"
            f"{'（演练中，不会真拦）' if pc.dry_run else ''}",
            f"配额：本小时撤回 {used}/{cfg.max_recalls_per_hour}"
            f" · 今日 {self._budget.recalls_today()}/{cfg.max_recalls_per_day}",
            f"审计：共 {self._audit.count()} 条，其中撤回 {self._audit.count('recalled')}"
            f" / 演练 {self._audit.count('dry_run')}"
            f" / 被拦 {self._audit.count('blocked')}",
            "指令：/air audit 查看流水 · /air test 就地试跑（管理员）",
        ]
        yield event.plain_result("\n".join(lines))

    async def _cmd_audit(self, event: AstrMessageEvent):
        records = self._audit.recent(8)
        if not records:
            yield event.plain_result(f"{TAG} 还没有任何判定记录。")
            return
        rows = [f"{TAG} 最近 {len(records)} 条判定"]
        for record in reversed(records):
            stamp = time.strftime("%m-%d %H:%M", time.localtime(record.get("ts", 0)))
            action = str(record.get("action") or "?")
            phase = "发" if record.get("phase") == "precheck" else "撤"
            mark = {
                "recalled": "✅撤回",
                "rewritten": "✍️重写",
                "blocked": "⛔拦截",
                "dry_run": "🧪演练",
                "kept": "➖保留",
                "failed": "❌失败",
            }.get(action, action)
            reason = str(record.get("reason", ""))
            if record.get("phase") == "precheck":
                reason = f"{record.get('draft', '')} → {reason}"
                rows.append(f"{stamp} [{phase}] {mark} {record.get('session', '')} {reason}")
                continue
            rows.append(
                f"{stamp} [{phase}] {mark} {record.get('session', '')} "
                f"{record.get('verdict', '')}"
                f"({float(record.get('confidence') or 0):.2f}) "
                f"{reason}"
            )
        yield event.plain_result("\n".join(rows))

    async def _cmd_test(self, event: AstrMessageEvent):
        if not event.is_admin():
            yield event.plain_result(f"{TAG} 只有管理员能试跑。")
            return
        trigger = self._collect(event)
        if trigger is None:
            yield event.plain_result(f"{TAG} 这个会话不在插件的处理范围内（或被过滤）。")
            return
        yield event.plain_result(f"{TAG} 正在试跑（不会真的撤回）…")
        result = await self._review(trigger, dry_run_override=True, force=True)
        if not result.ok:
            yield event.plain_result(f"{TAG} 没有可判断的对象：{result.stage}")
            return
        verdict = result.verdict or "未判定"
        head = [
            f"{TAG} 试跑结果：{verdict}"
            f"（置信度 {result.confidence:.2f}，动作 {result.action or '无'}）",
            f"理由：{result.reason or '—'}",
            f"目标行：{result.target_indexes}",
            "—— 聊天记录 ——",
            result.transcript or "（空）",
        ]
        yield event.plain_result("\n".join(head))
