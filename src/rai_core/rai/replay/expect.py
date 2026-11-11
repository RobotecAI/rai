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

"""Assertions for tests of agents and tools: what was sent, what was called.

Egress assertions take a :class:`ReplayConnector` (or an :class:`AgentReplay`,
or a list of :class:`EgressRecord`); tool assertions take an
:class:`AgentReplay`, a :class:`~rai.recorder.Recording` (or its path), or the
events an agent emitted (``agent.subscribe(events.append)``). Each returns
what matched and raises AssertionError listing what was seen otherwise.
"""

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from rai.agents.events import AgentEvent, ToolEnd, ToolStart
from rai.communication.ros2.egress import MOTION
from rai.recorder import TOOLS, Recording

from .connector import EgressRecord, ReplayConnector

Predicate = Union[Callable[[Any], bool], Mapping[str, Any], None]
"""A function of the message (or args), or ``{"field.path": value}``."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    status: Optional[str] = None  # success, error, denied; None if unfinished
    content: Any = None
    call_id: str = ""


def _field(value: Any, path: str) -> Any:
    for part in path.split("."):
        value = value[part] if isinstance(value, Mapping) else getattr(value, part)
    return value


def _matches(value: Any, predicate: Predicate) -> bool:
    if predicate is None:
        return True
    if callable(predicate):
        return bool(predicate(value))
    try:
        return all(_field(value, k) == v for k, v in predicate.items())
    except (AttributeError, KeyError, TypeError):
        return False


def _records(source: Any) -> tuple[list[EgressRecord], Optional[int]]:
    connector = getattr(source, "connector", source)
    if isinstance(connector, ReplayConnector):
        return list(connector.sent), connector.start
    return list(source), None


def _show(records: list[EgressRecord]) -> str:
    if not records:
        return "nothing was sent"
    lines = [f"  {r.kind} {r.name} ({r.type}) from {r.origin or '-'}" for r in records]
    return "sent:\n" + "\n".join(lines[:50])


def _expect_egress(
    source: Any,
    kind: str,
    name: str,
    predicate: Predicate,
    within: Optional[float],
    count: Optional[int],
) -> list[Any]:
    records, start = _records(source)
    name = name if name.startswith("/") else f"/{name}"
    on_name = [r for r in records if r.kind == kind and r.name == name]
    matched = [r for r in on_name if _matches(r.payload, predicate)]
    if within is not None:
        if start is None:
            raise ValueError("within needs a ReplayConnector (its clock)")
        late = start + int(within * 1e9)
        matched = [r for r in matched if r.stamp <= late]
    ok = len(matched) == count if count is not None else bool(matched)
    if not ok:
        wanted = f"{count if count is not None else 'at least one'} {kind}"
        when = f" within {within:g} s" if within is not None else ""
        raise AssertionError(
            f"expected {wanted} on {name}{when} matching {predicate!r},"
            f" found {len(matched)} ({len(on_name)} on {name} in all); "
            + _show(records)
        )
    return [r.payload for r in matched]


def expect_published(
    source: Any,
    topic: str,
    predicate: Predicate = None,
    *,
    within: Optional[float] = None,
    count: Optional[int] = None,
) -> list[Any]:
    """Messages published on topic matching predicate: at least one, or
    exactly count; with within, published at most within (virtual) seconds
    after the replay started."""
    return _expect_egress(source, "publish", topic, predicate, within, count)


def expect_called(
    source: Any,
    service: str,
    predicate: Predicate = None,
    *,
    within: Optional[float] = None,
    count: Optional[int] = None,
) -> list[Any]:
    """Requests sent to service matching predicate."""
    return _expect_egress(source, "service", service, predicate, within, count)


def expect_goal(
    source: Any,
    action: str,
    predicate: Predicate = None,
    *,
    within: Optional[float] = None,
    count: Optional[int] = None,
) -> list[Any]:
    """Goals sent to action matching predicate."""
    return _expect_egress(source, "goal", action, predicate, within, count)


def expect_nothing_sent(source: Any) -> None:
    """No publish, service call, goal or cancel at all."""
    records, _ = _records(source)
    if records:
        raise AssertionError("expected no egress; " + _show(records))


def expect_no_motion(source: Any, types: Iterable[str] = MOTION) -> None:
    """No egress of a motion type (MOTION by default): nothing that would
    have moved the robot. Cancels are allowed (they stop motion)."""
    types = frozenset(types)
    records, _ = _records(source)
    moving = [r for r in records if r.kind != "cancel" and r.type in types]
    if moving:
        raise AssertionError("expected no motion; " + _show(moving))


# ------------------------------------------------------------ tool calls


def tool_calls(source: Any) -> list[ToolCall]:
    """Tool calls of an AgentReplay, a Recording (or path) or agent events."""
    calls = getattr(source, "calls", None)
    if calls is not None and not callable(calls):
        return list(calls)
    if isinstance(source, (str, Path)):
        source = Recording(source)
    if isinstance(source, Recording):
        values = source.values(TOOLS)
        ends = {v["call_id"]: v for v in values if v["phase"] == "end"}
        out = []
        for v in values:
            if v["phase"] != "start":
                continue
            end = ends.get(v["call_id"], {})
            out.append(
                ToolCall(
                    v["name"],
                    v["args"],
                    end.get("status"),
                    end.get("content"),
                    v["call_id"],
                )
            )
        return out
    events: list[AgentEvent] = list(source)
    done = {e.call_id: e for e in events if isinstance(e, ToolEnd)}
    out = []
    for e in events:
        if isinstance(e, ToolStart):
            end = done.get(e.call_id)
            out.append(
                ToolCall(
                    e.name,
                    dict(e.args),
                    end.status if end else None,
                    end.content if end else None,
                    e.call_id,
                )
            )
    return out


def expect_tool_called(
    source: Any,
    name: str,
    args: Predicate = None,
    *,
    status: Optional[str] = None,
    count: Optional[int] = None,
) -> list[ToolCall]:
    """Calls of tool name whose args match (a predicate, or a dict of
    expected arguments) and, if given, that ended with status."""
    calls = tool_calls(source)
    matched = [
        c
        for c in calls
        if c.name == name
        and _matches(c.args, args)
        and (status is None or c.status == status)
    ]
    ok = len(matched) == count if count is not None else bool(matched)
    if not ok:
        seen = "\n".join(
            f"  {c.name}({json.dumps(c.args, default=repr)}) -> {c.status}"
            for c in calls
        )
        raise AssertionError(
            f"expected {count if count is not None else 'a'} call of {name}"
            f" with {args!r}{f' ending {status}' if status else ''},"
            f" found {len(matched)}; calls:\n{seen or '  none'}"
        )
    return matched
