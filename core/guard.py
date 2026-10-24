"""预算、冷却与审计日志。

撤回是不可逆的对外动作，所以这里把它当成「有配额的资源」来管。
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HOUR = 3600.0
DAY = 86400.0


def _prune(stamps: deque[float], now: float, window: float) -> None:
    while stamps and now - stamps[0] > window:
        stamps.popleft()


class RecallBudget:
    """每小时 / 每天 / 每会话的撤回配额，外加判定调用频控。"""

    def __init__(
        self,
        *,
        per_hour: int,
        per_day: int,
        session_cooldown: float,
        judgments_per_hour: int,
    ) -> None:
        self.per_hour = max(0, int(per_hour))
        self.per_day = max(0, int(per_day))
        self.session_cooldown = max(0.0, float(session_cooldown))
        self.judgments_per_hour = max(0, int(judgments_per_hour))
        self._recalls: deque[float] = deque(maxlen=512)
        self._judgments: deque[float] = deque(maxlen=4096)
        self._session_last: dict[str, float] = {}

    # -- 判定频控 ---------------------------------------------------------
    def allow_judgement(self, now: float) -> bool:
        _prune(self._judgments, now, HOUR)
        return len(self._judgments) < self.judgments_per_hour

    def note_judgement(self, now: float) -> None:
        self._judgments.append(now)

    # -- 撤回配额 ---------------------------------------------------------
    def allow_recall(self, session: str, now: float) -> tuple[bool, str]:
        _prune(self._recalls, now, DAY)
        if sum(1 for ts in self._recalls if now - ts <= HOUR) >= self.per_hour:
            return False, f"已达每小时上限 {self.per_hour} 次"
        if len(self._recalls) >= self.per_day:
            return False, f"已达每天上限 {self.per_day} 次"
        last = self._session_last.get(session, 0.0)
        if self.session_cooldown and now - last < self.session_cooldown:
            remain = int(self.session_cooldown - (now - last))
            return False, f"本会话冷却中，还需 {remain}s"
        return True, ""

    def note_recall(self, session: str, now: float) -> None:
        self._recalls.append(now)
        self._session_last[session] = now

    def recalls_last_hour(self, now: float) -> int:
        return sum(1 for ts in self._recalls if now - ts <= HOUR)

    def recalls_today(self) -> int:
        return len(self._recalls)

    # -- 持久化 -----------------------------------------------------------
    def to_state(self) -> dict[str, Any]:
        now = time.time()
        return {
            "recall_times": [ts for ts in self._recalls if now - ts <= DAY],
            "session_last": {
                key: ts for key, ts in self._session_last.items() if now - ts <= DAY
            },
        }

    def load_state(self, data: object) -> None:
        if not isinstance(data, dict):
            return
        now = time.time()
        for ts in data.get("recall_times") or []:
            try:
                value = float(ts)
            except (TypeError, ValueError):
                continue
            if now - value <= DAY:
                self._recalls.append(value)
        session_last = data.get("session_last")
        if isinstance(session_last, dict):
            for key, ts in session_last.items():
                try:
                    value = float(ts)
                except (TypeError, ValueError):
                    continue
                if now - value <= DAY:
                    self._session_last[str(key)] = value


class AuditLog:
    """判定与撤回的流水，既留在内存给指令看，也追加到 jsonl 落盘。"""

    def __init__(self, path: Path | None, keep: int = 300) -> None:
        self._path = path
        self._items: deque[dict[str, Any]] = deque(maxlen=max(10, keep))
        if path is not None:
            self._load_tail()

    def _load_tail(self) -> None:
        assert self._path is not None
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                for line in handle.readlines()[-self._items.maxlen :]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(record, dict):
                        self._items.append(record)
        except OSError as exc:
            logger.debug("读取审计日志失败: %s", exc)

    def add(self, record: dict[str, Any]) -> None:
        record = {"ts": time.time(), **record}
        self._items.append(record)
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.debug("写入审计日志失败: %s", exc)

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        return list(self._items)[-max(1, limit) :]

    def count(self, action: str | None = None) -> int:
        if action is None:
            return len(self._items)
        return sum(1 for item in self._items if item.get("action") == action)
