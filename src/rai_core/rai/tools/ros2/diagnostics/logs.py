# Copyright (C) 2026 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""/rosout and /diagnostics, buffered from the first call on."""

import asyncio
import threading
import time
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple, Type

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from pydantic import BaseModel, Field, PrivateAttr
from rcl_interfaces.msg import Log
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from rai.communication.ros2 import Subscription
from rai.tools.hooks import Effect
from rai.tools.ros2.base import BaseROS2Tool
from rai.tools.ros2.diagnostics.common import bounded, clip, now_s, seconds, stamp_s

# rcl's rosout publishers keep their last entries for late joiners (transient local).
ROSOUT_QOS = QoSProfile(
    depth=1000,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
LOG_LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40, "fatal": 50}
LOG_NAMES = {v: k.upper() for k, v in LOG_LEVELS.items()}
DIAGNOSTIC_LEVELS = {0: "OK", 1: "WARN", 2: "ERROR", 3: "STALE"}


def _int(value: Any) -> int:
    """uint8 fields are int, byte fields (DiagnosticStatus.level) bytes."""
    return value[0] if isinstance(value, bytes) else int(value)


LogLevel = Literal["debug", "info", "warn", "error", "fatal"]


class GetROS2LogsInput(BaseModel):
    min_level: LogLevel = Field("info", description="Lowest severity to include")
    node: Optional[str] = Field(
        None, description="Only loggers whose name contains this, e.g. amcl"
    )
    contains: Optional[str] = Field(None, description="Only messages containing this")
    since_s: Optional[float] = Field(
        None, gt=0, description="Only entries from the last since_s seconds"
    )
    last_n: int = Field(20, ge=1, le=200, description="At most this many, newest last")


class GetROS2LogsTool(BaseROS2Tool):
    """Tails /rosout. The subscription starts at the first call and keeps the
    last buffer_size entries; nodes' recent entries (rcl keeps them transient
    local) arrive with it."""

    default_effect = Effect.READ
    default_targets = ("/rosout",)
    name: str = "get_ros2_logs"
    description: str = (
        "Recent ROS 2 log entries from /rosout, filtered by level, node, text"
        " and time; ages are in seconds before now"
    )
    args_schema: Type[GetROS2LogsInput] = GetROS2LogsInput
    buffer_size: int = 2000
    warmup_s: float = 0.5

    _logs: Deque[Log] = PrivateAttr(default_factory=deque)
    _subscription: Optional[Subscription] = PrivateAttr(None)

    async def _acall(
        self,
        min_level: LogLevel = "info",
        node: Optional[str] = None,
        contains: Optional[str] = None,
        since_s: Optional[float] = None,
        last_n: int = 20,
    ) -> str:
        self.check_readable("/rosout")
        if self._subscription is None or self._subscription.closed:
            self._logs = deque(maxlen=self.buffer_size)
            self._subscription = self.connector.subscribe(
                "/rosout",
                self._logs.append,
                msg_type=Log,
                qos=ROSOUT_QOS,
                maxsize=self.buffer_size,
            )
            await asyncio.sleep(self.warmup_s)
        now = now_s(self)
        logger = node.strip("/").replace("/", ".") if node else None
        entries = [
            log
            for log in list(self._logs)
            if _int(log.level) >= LOG_LEVELS[min_level]
            and (logger is None or logger in log.name)
            and (contains is None or contains in log.msg)
            and (since_s is None or now - stamp_s(log.stamp) <= since_s)
        ]
        entries.sort(key=lambda log: stamp_s(log.stamp))
        levels = Counter(LOG_NAMES.get(_int(e.level), str(e.level)) for e in entries)
        header = (
            f"{len(entries)} matching entries"
            + (
                f" ({', '.join(f'{n} {k}' for k, n in levels.items())})"
                if levels
                else ""
            )
            + f", showing the last {min(last_n, len(entries))}"
            f" (buffer: {len(self._logs)} entries)"
        )
        lines = [
            f"-{seconds(now - stamp_s(e.stamp))} {LOG_NAMES.get(_int(e.level), e.level)}"
            f" [{e.name}] {clip(e.msg, 300)}"
            for e in entries[-last_n:]
        ]
        return "\n".join([header, *lines])


class GetROS2DiagnosticsInput(BaseModel):
    topic: str = Field(
        "/diagnostics",
        description="/diagnostics, or /diagnostics_agg for the aggregator's view",
    )
    stale_after_s: float = Field(
        5.0, gt=0, description="A status not updated for this long is STALE"
    )
    only_problems: bool = Field(False, description="Leave out OK statuses")


class _Statuses:
    """The latest status per (hardware id, name), with its arrival time."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest: Dict[Tuple[str, str], Tuple[DiagnosticStatus, float]] = {}
        self.subscription: Optional[Subscription] = None

    def add(self, msg: DiagnosticArray) -> None:
        received = time.monotonic()
        with self.lock:
            for status in msg.status:
                self.latest[(status.hardware_id, status.name)] = (status, received)

    def snapshot(self) -> List[Tuple[DiagnosticStatus, float]]:
        with self.lock:
            return list(self.latest.values())


class GetROS2DiagnosticsTool(BaseROS2Tool):
    """Summarizes diagnostic_msgs/DiagnosticArray statuses. The subscription
    starts at the first call for a topic and keeps the latest status per
    hardware id and name."""

    default_effect = Effect.READ
    target_args = {"topic": "topic"}
    name: str = "get_ros2_diagnostics"
    description: str = (
        "Summary of ROS 2 diagnostics: worst level overall and per hardware id,"
        " problems with their key values, and stale statuses"
    )
    args_schema: Type[GetROS2DiagnosticsInput] = GetROS2DiagnosticsInput
    warmup_s: float = 1.5  # diagnostic_updater publishes at 1 Hz by default

    _topics: Dict[str, _Statuses] = PrivateAttr(default_factory=dict)

    async def _acall(
        self,
        topic: str = "/diagnostics",
        stale_after_s: float = 5.0,
        only_problems: bool = False,
    ) -> str:
        self.check_readable(topic)
        topic = self.connector.resolve(topic)
        statuses = self._topics.setdefault(topic, _Statuses())
        if statuses.subscription is None or statuses.subscription.closed:
            statuses.subscription = self.connector.subscribe(
                topic, statuses.add, msg_type=DiagnosticArray
            )
            await asyncio.sleep(self.warmup_s)
        now = time.monotonic()
        rows = []  # (level, line, hardware id)
        for status, received in statuses.snapshot():
            level, age = _int(status.level), now - received
            hardware = status.hardware_id or status.name
            text = f"{status.name}" + (
                f" [hw {status.hardware_id}]" if status.hardware_id else ""
            )
            if age > stale_after_s:
                text += f": no update for {seconds(age)} (last {DIAGNOSTIC_LEVELS.get(level, level)}: {clip(status.message, 80)})"
                level = 3
            elif level:
                values = ", ".join(f"{kv.key}={kv.value}" for kv in status.values[:5])
                text += f": {clip(status.message, 120)}" + (
                    f" | {clip(values, 160)}" if values else ""
                )
            rows.append((level, text, hardware))
        if not rows:
            publishers = len(self.connector.get_topic_endpoints(topic)[0])
            return f"No statuses on {topic} yet (publishers: {publishers})"
        rows.sort(key=lambda r: (-r[0], r[1]))
        levels = Counter(DIAGNOSTIC_LEVELS.get(r[0], str(r[0])) for r in rows)
        worst: Dict[str, int] = {}
        for level, _, hardware in rows:
            worst[hardware] = max(worst.get(hardware, 0), level)
        lines = [
            f"{topic}: {len(rows)} statuses, worst {DIAGNOSTIC_LEVELS.get(rows[0][0])}"
            f" ({', '.join(f'{n} {k}' for k, n in levels.items())})"
        ]
        lines += bounded(
            (
                f"{DIAGNOSTIC_LEVELS.get(level, level)} {text}"
                for level, text, _ in rows
                if level or not only_problems
            ),
            60,
        )
        lines.append(
            "Worst per hardware id: "
            + ", ".join(
                f"{hw} {DIAGNOSTIC_LEVELS.get(level)}"
                for hw, level in sorted(worst.items(), key=lambda i: -i[1])[:30]
            )
        )
        return "\n".join(lines)
