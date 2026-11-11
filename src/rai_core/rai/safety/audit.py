# Copyright (C) 2025 Robotec.AI
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

"""AuditLog: a record of every hook decision on both hook points.

AuditLog is an observer, not a hook: it is registered with ``observe()`` on
the tool and egress chains, so it cannot allow or deny anything, and its own
failures (a full queue, a full disk, a failing subscriber) never block or
deny an action. They are counted in ``dropped`` and logged.
"""

import dataclasses
import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Self, Union

from rai.communication.ros2.egress import EgressDecision
from rai.origin import egress_origin
from rai.tools.hooks import ToolDecision, ToolHooks, global_tool_hooks

from ._base import Target, egress_chains

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditRecord:
    """One hook decision. ``to_dict()`` is what the JSONL file holds."""

    time: float  # wall clock, seconds since the epoch
    side: Literal["tool", "egress"]
    origin: Optional[str]  # the tool (or caller tag) that caused it, if any
    kind: str  # "tool", or the egress kind: publish, service, goal, cancel
    name: str  # the tool name, or the resolved topic/service/action name
    type: str  # the tool's effect, or the ROS type ("pkg/msg/Name")
    decision: Literal["allow", "deny"]
    hook: str
    reason: str = ""
    lease_s: float = 0.0  # lease granted by this decision
    leased: bool = False  # allowed by an earlier lease; the hook did not run
    targets: tuple[str, ...] = ()  # tool side: the call's ROS targets
    args: Mapping[str, Any] = field(default_factory=dict)  # tool side
    payload: Any = field(default=None, compare=False)  # egress side; not in JSON

    def to_dict(self) -> dict[str, Any]:
        record = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("payload", "args", "targets")
        }
        if self.side == "tool":
            record["targets"] = list(self.targets)
            record["args"] = dict(self.args)
        return record


class AuditLog:
    """Writes every tool and egress decision as JSON lines to ``path`` (none:
    no file) and passes each AuditRecord to subscribers.

    Records are queued (``maxsize``) and written by one background thread, so
    the hook chains never wait on the disk; when the queue is full records are
    dropped and counted. Subscribers run on that thread, in order.
    """

    def __init__(self, path: Union[str, Path, None] = None, *, maxsize: int = 10_000):
        self.path = Path(path) if path is not None else None
        self.dropped = 0
        self._file = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("a", encoding="utf-8")
        self._queue: queue.Queue[Optional[AuditRecord]] = queue.Queue(maxsize)
        self._subscribers: tuple[Callable[[AuditRecord], Any], ...] = ()
        self._installed: list[tuple[Any, Callable[[Any], None]]] = []
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="rai_audit", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ inputs

    def install(
        self, *targets: Target, tools: Optional[ToolHooks] = global_tool_hooks
    ) -> Self:
        """Observe the egress chains of targets (every runtime if none) and the
        tool chain tools (every RaiTool by default; None: connector side only)."""
        for chain in egress_chains(targets):
            chain.observe(self.on_egress)
            self._installed.append((chain, self.on_egress))
        if tools is not None:
            tools.observe(self.on_tool)
            self._installed.append((tools, self.on_tool))
        return self

    def uninstall(self) -> None:
        for chain, observer in self._installed:
            chain.unobserve(observer)
        self._installed.clear()

    def on_tool(self, decision: ToolDecision) -> None:
        """Record an agent-side decision. Never raises."""
        try:
            call = decision.call
            self._put(
                AuditRecord(
                    time.time(),
                    "tool",
                    egress_origin.get(),
                    "tool",
                    call.name,
                    str(call.effect),
                    "allow" if decision.allowed else "deny",
                    decision.hook,
                    decision.reason,
                    targets=tuple(call.targets),
                    args=call.args,
                )
            )
        except Exception:
            self._failed("tool decision")

    def on_egress(self, decision: EgressDecision) -> None:
        """Record a connector-side decision. Never raises."""
        try:
            egress = decision.egress
            self._put(
                AuditRecord(
                    time.time(),
                    "egress",
                    egress.origin,
                    egress.kind,
                    egress.name,
                    egress.type,
                    "allow" if decision.allowed else "deny",
                    decision.hook,
                    decision.reason,
                    lease_s=decision.lease_s,
                    leased=decision.leased,
                    payload=egress.payload,
                )
            )
        except Exception:
            self._failed("egress decision")

    def _put(self, record: AuditRecord) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._failed("record (queue full)", log=self.dropped == 0)

    def _failed(self, what: str, log: bool = True) -> None:
        self.dropped += 1
        if log:
            logger.exception("AuditLog dropped a %s", what)

    # ------------------------------------------------------------ outputs

    def subscribe(self, callback: Callable[[AuditRecord], Any]) -> Callable[[], None]:
        """Call callback(record) for every record (for the flight recorder).
        Returns a function that unsubscribes. Errors are logged."""
        self._subscribers = (*self._subscribers, callback)

        def unsubscribe() -> None:
            self._subscribers = tuple(s for s in self._subscribers if s is not callback)

        return unsubscribe

    def _run(self) -> None:
        while True:
            record = self._queue.get()
            try:
                if record is None:
                    return
                self._write(record)
                for subscriber in self._subscribers:
                    try:
                        subscriber(record)
                    except Exception:
                        logger.exception("AuditLog subscriber %r failed", subscriber)
            finally:
                self._queue.task_done()

    def _write(self, record: AuditRecord) -> None:
        if self._file is None:
            return
        try:
            self._file.write(json.dumps(record.to_dict(), default=repr) + "\n")
            self._file.flush()
        except Exception:
            self._failed("record (write failed)")

    def flush(self) -> None:
        """Wait until every queued record is written and delivered."""
        if self._thread.is_alive():
            self._queue.join()

    def close(self) -> None:
        """Stop observing, write what is queued and close the file."""
        if self._closed:
            return
        self.uninstall()
        self._closed = True
        try:
            self._queue.put(None, timeout=5.0)
            self._thread.join(5.0)
        except queue.Full:
            logger.error("AuditLog writer is stuck; closing without it")
        if self._file is not None:
            self._file.close()
