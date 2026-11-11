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

# The classes of this module use each other's private members.
# pyright: reportPrivateUsage=false

"""Connector-side egress hooks: a fail-closed chain run by the runtime before
every publish, service call, goal and goal cancel (rai3-plan/PLAN.md 2.3).

An `EgressHook` is `(Egress) -> Allow | Deny`, sync or async. It declares what it
matches (types first, since names get remapped; then resolved-name patterns;
then kinds), so egress it does not match pays nothing. A hook that raises,
times out or returns anything else denies, and a denial raises `EgressDenied`
in the caller. `Allow(lease_s=...)` approves the same (kind, name, type) for that
hook for a while, so a 30 Hz stream is not re-approved per message; other hooks
still run. Agent-side tool hooks (`rai.tools.hooks`) are a separate chain.

Hooks run on the caller's event loop (async API) or the runtime's private loop
(sync API), never on the executor thread. Sync hooks run inline and cannot be
timed out: keep them fast, or make them async.
"""

import asyncio
import functools
import inspect
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Literal, NoReturn, Optional, Union

from rai.origin import egress_origin

logger = logging.getLogger(__name__)

DEFAULT_EGRESS_TIMEOUT_S = 2.0

EgressKind = Literal["publish", "service", "goal", "cancel"]
ALL_KINDS: frozenset[str] = frozenset({"publish", "service", "goal", "cancel"})
WRITE_KINDS: frozenset[str] = frozenset({"publish", "service", "goal"})
"""The default kinds of a hook. Cancels stop motion, so a hook gates them only
when it asks for them (kinds=ALL_KINDS): a confirmation or e-stop hook must never
hold back a cancel by accident."""

MOTION: frozenset[str] = frozenset(
    {
        "geometry_msgs/msg/Twist",
        "geometry_msgs/msg/TwistStamped",
        "trajectory_msgs/msg/JointTrajectory",
        "control_msgs/action/FollowJointTrajectory",
        "control_msgs/action/GripperCommand",
        "nav2_msgs/action/NavigateToPose",
        "nav2_msgs/action/NavigateThroughPoses",
        "nav2_msgs/action/FollowPath",
        "nav2_msgs/action/FollowWaypoints",
        "nav2_msgs/action/FollowGPSWaypoints",
        "nav2_msgs/action/Spin",
        "nav2_msgs/action/BackUp",
        "nav2_msgs/action/DriveOnHeading",
        "nav2_msgs/action/AssistedTeleop",
        "moveit_msgs/action/MoveGroup",
        "moveit_msgs/action/ExecuteTrajectory",
        "moveit_msgs/srv/ExecuteKnownTrajectory",
        "rai_interfaces/srv/ManipulatorMoveTo",
    }
)
"""Message, service and action types that move hardware. Extend it with a union:
``MOTION | {"my_msgs/msg/WheelSpeeds"}``."""


@dataclass(frozen=True)
class Egress:
    kind: str  # one of ALL_KINDS
    name: str  # resolved topic, service or action name
    type: str  # "pkg/msg/Name", "pkg/srv/Name" or "pkg/action/Name"
    payload: Any  # message, request, goal, or the goal id of a cancel (read-only)
    origin: Optional[str] = None


@dataclass(frozen=True)
class Allow:
    lease_s: float = 0.0


@dataclass(frozen=True)
class Deny:
    reason: str


EgressHook = Callable[[Egress], Union[Allow, Deny, Awaitable[Union[Allow, Deny]]]]


class EgressDenied(PermissionError):
    """An egress hook refused a publish, service call, goal or cancel."""

    def __init__(self, egress: Egress, hook: str, reason: str) -> None:
        self.egress = egress
        self.hook = hook
        self.reason = reason
        super().__init__(
            f"{egress.kind} {egress.type} on {egress.name} denied by {hook}: {reason}"
        )


@dataclass(frozen=True)
class EgressDecision:
    """What observers see: one hook's verdict on one egress."""

    egress: Egress
    hook: str
    allowed: bool
    reason: str = ""
    leased: bool = False  # allowed by an earlier lease; the hook did not run
    lease_s: float = 0.0  # the lease this decision granted


Key = tuple[str, str, str]  # kind, resolved name, type


@dataclass(eq=False)
class _Entry:
    hook: EgressHook
    name: str
    types: frozenset[str]
    names: tuple[str, ...]
    kinds: frozenset[str]
    timeout_s: float
    leases: dict[Key, float] = field(default_factory=dict[Key, float])

    def matches(self, key: Key) -> bool:
        kind, name, type_ = key
        if kind not in self.kinds:
            return False
        if not self.types and not self.names:
            return True
        return type_ in self.types or any(fnmatchcase(name, p) for p in self.names)


class EgressHooks:
    """An ordered egress hook chain. Hooks run in insertion order; the first
    denial wins. Matches are cached per (kind, name, type)."""

    def __init__(self) -> None:
        self._entries: tuple[_Entry, ...] = ()
        self._cache: dict[Key, tuple[_Entry, ...]] = {}
        self._observers: tuple[Callable[[EgressDecision], None], ...] = ()
        self._lock = threading.Lock()

    def add(
        self,
        hook: EgressHook,
        *,
        types: Optional[Iterable[str]] = None,
        names: Optional[Iterable[str]] = None,
        kinds: Iterable[str] = WRITE_KINDS,
        timeout_s: float = DEFAULT_EGRESS_TIMEOUT_S,
    ) -> EgressHook:
        """Run hook on egress whose type is in types or whose resolved name
        matches a pattern in names (fnmatch, e.g. "/*/cmd_vel"), restricted to
        kinds. With neither types nor names it sees every egress of its kinds."""
        kinds = frozenset(kinds)
        if not kinds <= ALL_KINDS:
            raise ValueError(f"Unknown egress kinds {set(kinds - ALL_KINDS)}")
        entry = _Entry(
            hook,
            getattr(hook, "__name__", type(hook).__name__),
            frozenset(types or ()),
            tuple(names or ()),
            kinds,
            timeout_s,
        )
        with self._lock:
            self._entries = (*self._entries, entry)
            self._cache = {}
        return hook

    def remove(self, hook: EgressHook) -> None:
        """Remove hook (every registration of it) and its leases."""
        with self._lock:
            self._entries = tuple(e for e in self._entries if e.hook is not hook)
            self._cache = {}

    def clear(self) -> None:
        with self._lock:
            self._entries = ()
            self._cache = {}

    def revoke(self, hook: Optional[EgressHook] = None) -> None:
        """Drop the leases of hook, or of every hook: the next egress asks again."""
        for entry in self._entries:
            if hook is None or entry.hook is hook:
                entry.leases.clear()

    def observe(
        self, observer: Callable[[EgressDecision], None]
    ) -> Callable[[EgressDecision], None]:
        """Call observer(decision) for every hook verdict of this chain, leases
        included. It runs inline: keep it fast. Its errors are logged."""
        with self._lock:
            self._observers = (*self._observers, observer)
        return observer

    def unobserve(self, observer: Callable[[EgressDecision], None]) -> None:
        with self._lock:
            self._observers = tuple(o for o in self._observers if o != observer)

    def __len__(self) -> int:
        return len(self._entries)

    def _match(self, key: Key) -> tuple[_Entry, ...]:
        cache = self._cache
        entries = cache.get(key)
        if entries is None:
            entries = cache[key] = tuple(e for e in self._entries if e.matches(key))
        return entries


global_egress = EgressHooks()
"""Hooks run before the hooks of every runtime, the shared one included. They
survive the shared runtime being closed and re-created."""


@functools.cache
def type_name(cls: type) -> str:
    """ "geometry_msgs/msg/Twist" for the Twist class (also srv and action)."""
    package, kind = cls.__module__.split(".")[:2]
    return f"{package}/{kind}/{cls.__name__}"


def _notify(chain: EgressHooks, decision: EgressDecision) -> None:
    logger.debug(
        "egress %s %s on %s: %s by %s%s%s",
        decision.egress.kind,
        decision.egress.type,
        decision.egress.name,
        "allowed" if decision.allowed else "denied",
        decision.hook,
        " (lease)" if decision.leased else "",
        f": {decision.reason}" if decision.reason else "",
    )
    for observer in chain._observers:
        try:
            observer(decision)
        except Exception:
            logger.exception("Egress observer %r failed", observer)


class EgressCheck:
    """One egress run through the matched hooks of a set of chains.

    step() runs sync hooks until the chain is done (returns None) or a hook
    returns an awaitable (returned to the caller, who passes it to finish()).
    """

    def __init__(
        self,
        matched: list[tuple[EgressHooks, _Entry]],
        key: Key,
        payload: Any,
    ) -> None:
        self._egress: Optional[Egress] = None
        self._payload = payload
        self._origin = egress_origin.get()
        self._key = key
        self._matched = matched
        self._i = 0

    @property
    def egress(self) -> Egress:
        if self._egress is None:  # not built when every hook holds a lease
            self._egress = Egress(*self._key, self._payload, self._origin)
        return self._egress

    def step(self) -> Optional[Awaitable[Any]]:
        while self._i < len(self._matched):
            chain, entry = self._matched[self._i]
            if entry.leases.get(self._key, 0.0) > time.monotonic():
                self._i += 1
                if chain._observers or logger.isEnabledFor(logging.DEBUG):
                    _notify(
                        chain,
                        EgressDecision(self.egress, entry.name, True, leased=True),
                    )
                continue
            try:
                decision = entry.hook(self.egress)
            except Exception as e:  # fail closed on any hook error
                self._deny(f"failed ({type(e).__name__}: {e})")
            if inspect.isawaitable(decision):
                return decision
            self._apply(decision)
        return None

    async def finish(self, pending: Optional[Awaitable[Any]]) -> None:
        while pending is not None:
            entry = self._matched[self._i][1]
            try:
                async with asyncio.timeout(entry.timeout_s):
                    decision = await pending
            except TimeoutError:
                self._deny(f"timed out after {entry.timeout_s:g} s")
            except Exception as e:  # fail closed on any hook error
                self._deny(f"failed ({type(e).__name__}: {e})")
            self._apply(decision)
            pending = self.step()

    def refuse(self, reason: str, pending: Optional[Awaitable[Any]] = None) -> NoReturn:
        """Deny without running the current hook (it cannot run here)."""
        if inspect.iscoroutine(pending):
            pending.close()
        self._deny(reason)

    def _apply(self, decision: Any) -> None:
        chain, entry = self._matched[self._i]
        if isinstance(decision, Allow):
            if decision.lease_s > 0:
                entry.leases[self._key] = time.monotonic() + decision.lease_s
            self._i += 1
            if chain._observers or logger.isEnabledFor(logging.DEBUG):
                _notify(
                    chain,
                    EgressDecision(
                        self.egress, entry.name, True, lease_s=decision.lease_s
                    ),
                )
            return
        if isinstance(decision, Deny):
            self._deny(decision.reason, hook_failed=False)
        self._deny(f"returned {decision!r}, expected Allow or Deny")

    def _deny(self, reason: str, hook_failed: bool = True) -> NoReturn:
        chain, entry = self._matched[self._i]
        if hook_failed:
            logger.warning(
                "Egress %s on %s denied: hook %s %s",
                self.egress.kind,
                self.egress.name,
                entry.name,
                reason,
            )
        _notify(chain, EgressDecision(self.egress, entry.name, False, reason))
        raise EgressDenied(self.egress, entry.name, reason)


def match(chains: Iterable[EgressHooks], key: Key) -> list[tuple[EgressHooks, _Entry]]:
    return [(chain, entry) for chain in chains for entry in chain._match(key)]


__all__ = [
    "ALL_KINDS",
    "DEFAULT_EGRESS_TIMEOUT_S",
    "MOTION",
    "WRITE_KINDS",
    "Allow",
    "Deny",
    "Egress",
    "EgressDecision",
    "EgressDenied",
    "EgressHook",
    "EgressHooks",
    "egress_origin",
    "global_egress",
    "type_name",
]
