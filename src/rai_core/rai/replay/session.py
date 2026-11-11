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

"""AgentReplay: re-run a recorded agent session offline and report where it
departs from the recording."""

import bisect
import json
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from rclpy.serialization import deserialize_message

from rai.agents.base import BaseAgent
from rai.agents.events import AgentEvent, ModelStart, Observation, ToolEnd, ToolStart
from rai.communication.ros2.egress import MOTION
from rai.communication.ros2.runtime import message_class
from rai.recorder import LLM, OBSERVATIONS, TOOLS, Recording
from rai.recorder.encode import prepare, ref_of

from .connector import EgressRecord, ReplayConnector, Response
from .expect import ToolCall
from .llm import ReplayChatModel, ReplayDivergence, ReplayExhausted, recorded_agent


@dataclass(frozen=True)
class Divergence:
    """One place where the replay departed from the recording."""

    kind: str  # prompt, observation, tool_call, tool_result, egress, model_calls, error
    where: str  # "model call 2", "tool drive (call_ab12)", ...
    detail: str
    expected: Any = None
    actual: Any = None

    def __str__(self) -> str:
        return f"{self.kind} at {self.where}: {self.detail}"


@dataclass
class ReplayReport:
    """What a replayed run did and where it diverged."""

    divergences: list[Divergence]
    model_calls: int
    recorded_model_calls: int
    tool_calls: list[tuple[str, dict[str, Any]]]
    recorded_tool_calls: list[tuple[str, dict[str, Any]]]
    sent: list[EgressRecord]
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.divergences

    def __str__(self) -> str:
        head = (
            f"replay {'matches the recording' if self.ok else 'DIVERGED'}:"
            f" {self.model_calls}/{self.recorded_model_calls} model calls,"
            f" {len(self.tool_calls)}/{len(self.recorded_tool_calls)} tool calls,"
            f" {len(self.sent)} egress"
        )
        lines = [head, *(f"  - {d}" for d in self.divergences)]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    def raise_on_divergence(self) -> None:
        if not self.ok:
            raise ReplayDivergence(str(self), self.divergences)


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=repr))


def _ref(kind: str, fmt: str, data: str) -> str:
    return ref_of(data)


def _prepared(value: Any) -> Any:
    """value as the recorder stores it (media as tokens), JSON round-tripped."""
    return _json(prepare(value, _ref))


def _essential(message: Mapping[str, Any]) -> dict[str, Any]:
    """The fields of a LangChain message dict that make the prompt: no ids,
    metadata, usage or timings."""
    data = message.get("data", {})
    out: dict[str, Any] = {"type": message.get("type"), "content": data.get("content")}
    calls = data.get("tool_calls") or []
    if calls:
        out["tool_calls"] = [
            {"name": c.get("name"), "args": c.get("args"), "id": c.get("id")}
            for c in calls
        ]
    for key in ("tool_call_id", "name"):
        if data.get(key):
            out[key] = data[key]
    return out


def _short(value: Any, limit: int = 200) -> str:
    text = json.dumps(value, ensure_ascii=False, default=repr)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _diff_prompt(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> str:
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e != a:
            keys = [k for k in sorted(set(e) | set(a)) if e.get(k) != a.get(k)]
            parts = [
                f"{k}: expected {_short(e.get(k))}, got {_short(a.get(k))}"
                for k in keys
            ]
            return f"message {i} ({e.get('type')}) differs: " + "; ".join(parts)
    if len(expected) != len(actual):
        return f"{len(actual)} messages, expected {len(expected)}"
    return "differs"


def _full_prompts(requests: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The full prompt of every request (the recorder stores deltas)."""
    out, previous = [], []
    for request in requests:
        full = previous[: request["prefix"]] + request["messages"]
        out.append(full)
        previous = full
    return out


def _to_dict(msg: Any) -> Any:
    from rai.communication.ros2.api.conversion import ros2_message_to_dict

    try:
        return ros2_message_to_dict(msg)
    except Exception:
        return repr(msg)


class AgentReplay:
    """Re-run one agent of a flight recording offline, deterministically.

    It gives the agent's parts and checks the run against the recording:

    - :attr:`model`: a :class:`ReplayChatModel` with the recorded responses.
    - :attr:`connector`: a :class:`ReplayConnector` on the recorded topics.
      Its clock follows the session: it moves to the recorded time of each
      model call and tool call, so tools read what the original tools could.
    - :meth:`attach` (or :meth:`run`) compares, as the agent works, every
      prompt (modulo ids, metadata and usage), StateBasedAgent observation,
      tool call and tool result with the recording, and at the end every
      message a tool published on a recorded topic with what the original
      tool sent (recorded messages are attributed to the latest call started
      before them). With ``strict`` the first divergence
      fails the next model call, so the run stops there.

    ::

        replay = AgentReplay("run.mcap")
        agent = ReActAgent(replay.model, make_tools(replay.connector), name="pilot")
        report = await replay.run(agent)
        assert report.ok, report
    """

    def __init__(
        self,
        recording: Union[str, Path, Recording],
        agent: Optional[str] = None,
        *,
        services: Optional[Mapping[str, Response]] = None,
        actions: Optional[Mapping[str, Response]] = None,
        strict: bool = True,
        egress_topics: Iterable[str] = (),
        egress_grace_s: float = 1.0,
    ) -> None:
        """
        Args:
            recording: a flight recording (F2) of the session.
            agent: the agent to replay (default: the only one with model calls).
            services, actions: scripted responses (see ReplayConnector).
            strict: stop the run at the first divergence.
            egress_topics: recorded topics to compare egress on, besides those
                the replay published to and those of a MOTION type.
            egress_grace_s: how long after a tool call ended a recorded
                message on an egress topic still counts as sent by it.
        """
        if not isinstance(recording, Recording):
            recording = Recording(recording)
        self.recording = recording
        self.agent_name = recorded_agent(recording, agent)
        self.strict = strict
        self.model = ReplayChatModel.from_recording(recording, self.agent_name)
        self.connector = ReplayConnector(recording, services=services, actions=actions)
        name = self.agent_name
        llm = [m for m in recording.messages([LLM]) if m.value["agent"] == name]
        requests = [m for m in llm if m.value["phase"] == "request"]
        self._request_stamps = [m.log_time for m in requests]
        self._requests = [m.value for m in requests]
        self._prompts = [
            [_essential(d) for d in p]
            for p in _full_prompts([r for r in self._requests])
        ]
        tools = [m for m in recording.messages([TOOLS]) if m.value["agent"] == name]
        self._starts = {
            m.value["call_id"]: m for m in tools if m.value["phase"] == "start"
        }
        self._ends = {m.value["call_id"]: m for m in tools if m.value["phase"] == "end"}
        self._observations = [
            m.value["values"]
            for m in recording.messages([OBSERVATIONS])
            if m.value["agent"] == name
        ]
        self._egress_topics = set(egress_topics)
        self.egress_grace_s = egress_grace_s
        self._lock = threading.Lock()
        self.divergences: list[Divergence] = []
        self._model_calls = 0
        self._observed = 0
        self.calls: list[ToolCall] = []  # the replay's tool calls
        self._open: dict[str, int] = {}
        self._egress: dict[str, list[EgressRecord]] = {}

    # ------------------------------------------------------------ inputs

    def turns(self) -> list[list[BaseMessage]]:
        """The new messages of every recorded turn, in order: what to pass
        to ``agent.ainvoke`` to repeat the session."""
        full = _full_prompts(self._requests)
        out: list[list[BaseMessage]] = []
        previous_last: Optional[list[Any]] = None
        seen: list[str] = []
        for request, prompt in zip(self._requests, full):
            turn = request.get("turn", "")
            if turn in seen:
                previous_last = prompt
                continue
            seen.append(turn)
            if previous_last is None:
                skip = 1 if prompt and prompt[0].get("type") == "system" else 0
            else:
                skip = len(previous_last) + 1  # the last prompt and its answer
            out.append(messages_from_dict(self.recording.restore(prompt[skip:])))
            previous_last = prompt
        return out

    @property
    def recorded_tool_calls(self) -> list[tuple[str, dict[str, Any]]]:
        return [(m.value["name"], m.value["args"]) for m in self._starts.values()]

    # ------------------------------------------------------------ checking

    def attach(self, agent: BaseAgent) -> Callable[[], None]:
        """Follow and check agent's events; returns the unsubscribe function."""
        return agent.subscribe(self._on_event)

    async def run(self, agent: BaseAgent) -> ReplayReport:
        """Attach to agent, replay every recorded turn (``ainvoke``) and
        return the report. The run stops at the first error."""
        unsubscribe = self.attach(agent)
        try:
            for i, messages in enumerate(self.turns()):
                try:
                    await agent.ainvoke(*messages)  # type: ignore[attr-defined]
                except ReplayExhausted as e:
                    self._diverge(Divergence("model_calls", f"turn {i + 1}", str(e)))
                    break
                except ReplayDivergence:
                    break  # already reported
                except Exception as e:
                    detail = f"the turn failed: {type(e).__name__}: {e}"
                    self._diverge(Divergence("error", f"turn {i + 1}", detail))
                    break
        finally:
            unsubscribe()
        return self.report()

    def _diverge(self, divergence: Divergence) -> None:
        with self._lock:
            first = not self.divergences
            self.divergences.append(divergence)
        if self.strict and first:
            self.model.fail_next(ReplayDivergence(str(divergence), [divergence]))

    def _on_event(self, event: AgentEvent) -> None:
        if event.agent != self.agent_name:
            return
        if isinstance(event, ModelStart):
            self._on_model(event)
        elif isinstance(event, Observation):
            self._on_observation(event)
        elif isinstance(event, ToolStart):
            self._on_tool_start(event)
        elif isinstance(event, ToolEnd):
            self._on_tool_end(event)

    def _on_model(self, event: ModelStart) -> None:
        k = self._model_calls
        self._model_calls += 1
        if k >= len(self._prompts):
            return  # the model raises ReplayExhausted
        self.connector.advance_to(self._request_stamps[k])
        actual = [_essential(_prepared(message_to_dict(m))) for m in event.messages]
        expected = _json(self._prompts[k])
        if actual != expected:
            detail = _diff_prompt(expected, actual)
            self._diverge(
                Divergence("prompt", f"model call {k + 1}", detail, expected, actual)
            )

    def _on_observation(self, event: Observation) -> None:
        k = self._observed
        self._observed += 1
        if k >= len(self._observations):
            detail = "an observation the recording does not have"
            self._diverge(Divergence("observation", f"observation {k + 1}", detail))
            return
        expected = _json(self._observations[k])
        actual = _prepared(event.values)
        if actual != expected:
            keys = sorted(
                n
                for n in set(expected) | set(actual)
                if expected.get(n) != actual.get(n)
            )
            detail = "; ".join(
                f"{n}: expected {_short(expected.get(n))}, got {_short(actual.get(n))}"
                for n in keys
            )
            self._diverge(
                Divergence(
                    "observation", f"observation {k + 1}", detail, expected, actual
                )
            )

    def _on_tool_start(self, event: ToolStart) -> None:
        self.calls.append(ToolCall(event.name, dict(event.args), call_id=event.call_id))
        self._open[event.call_id] = len(self.connector.sent)
        where = f"tool {event.name} ({event.call_id})"
        recorded = self._starts.get(event.call_id)
        if recorded is None:
            detail = "a tool call the recording does not have"
            self._diverge(Divergence("tool_call", where, detail))
            return
        self.connector.advance_to(recorded.log_time)
        expected = (recorded.value["name"], recorded.value["args"])
        actual = (event.name, _prepared(event.args))
        if actual != expected:
            detail = f"expected {_short(expected)}, got {_short(actual)}"
            self._diverge(Divergence("tool_call", where, detail, expected, actual))

    def _on_tool_end(self, event: ToolEnd) -> None:
        start = self._open.pop(event.call_id, len(self.connector.sent))
        for i, call in enumerate(self.calls):
            if call.call_id == event.call_id:
                self.calls[i] = ToolCall(
                    call.name, call.args, event.status, event.content, call.call_id
                )
        self._egress[event.call_id] = [
            r for r in self.connector.sent[start:] if r.origin in (None, event.name)
        ]
        recorded = self._ends.get(event.call_id)
        if recorded is None:
            return  # reported at the start
        where = f"tool {event.name} ({event.call_id})"
        expected = {
            "status": recorded.value["status"],
            "content": recorded.value["content"],
        }
        actual = {"status": event.status, "content": _prepared(event.content)}
        if actual != expected:
            keys = [k for k in ("status", "content") if actual[k] != expected[k]]
            detail = "; ".join(
                f"{k}: expected {_short(expected[k])}, got {_short(actual[k])}"
                for k in keys
            )
            self._diverge(Divergence("tool_result", where, detail, expected, actual))

    # ------------------------------------------------------------ report

    def _recorded_egress(self, topic: str) -> dict[str, list[Any]]:
        """Recorded messages on topic by the tool call that sent them: the
        latest call of the agent started before the message was received, if
        it arrived at most ``egress_grace_s`` after that call ended (a message
        is stamped on arrival, which can be after its tool returned)."""
        calls = sorted((m.log_time, call_id) for call_id, m in self._starts.items())
        if not calls:
            return {}
        grace = int(self.egress_grace_s * 1e9)
        cls = message_class(self.connector.bag.topics[topic])
        out: dict[str, list[Any]] = {}
        stamps = [t for t, _ in calls]
        for m in self.connector.bag.messages([topic], start=stamps[0]):
            call_id = calls[bisect.bisect_right(stamps, m.stamp) - 1][1]
            end = self._ends.get(call_id)
            if end is not None and m.stamp > end.log_time + grace:
                continue
            out.setdefault(call_id, []).append(deserialize_message(m.data, cls))
        return out

    def _check_egress(self, notes: list[str]) -> list[Divergence]:
        topics = self.connector.bag.topics
        compared = self._egress_topics | {
            t for t, type_ in topics.items() if type_ in MOTION
        }
        for records in self._egress.values():
            compared |= {r.name for r in records if r.kind == "publish"}
        compared &= set(topics)
        recorded = {topic: self._recorded_egress(topic) for topic in compared}
        out = []
        uncompared = set()
        for call_id, start in self._starts.items():
            if call_id not in self._egress:
                continue  # not run (reported elsewhere)
            records = self._egress[call_id]
            uncompared |= {
                f"{r.kind} {r.name}"
                for r in records
                if r.kind != "publish" or r.name not in compared
            }
            where = f"tool {start.value['name']} ({call_id})"
            for topic in sorted(compared):
                expected = recorded[topic].get(call_id, [])
                actual = [
                    r.payload
                    for r in records
                    if r.kind == "publish" and r.name == topic
                ]
                if actual != expected:
                    e = [_to_dict(m) for m in expected]
                    a = [_to_dict(m) for m in actual]
                    detail = (
                        f"published on {topic}: expected {_short(e)}, got {_short(a)}"
                    )
                    out.append(Divergence("egress", where, detail, expected, actual))
        if uncompared:
            notes.append(
                "not compared (no recorded counterpart): "
                + ", ".join(sorted(uncompared))
            )
        return out

    def report(self) -> ReplayReport:
        """The report of the run so far (call after the run)."""
        notes: list[str] = []
        divergences = list(self.divergences)
        divergences += self._check_egress(notes)
        missing = [
            f"{m.value['name']} ({call_id})"
            for call_id, m in self._starts.items()
            if call_id not in self._egress and call_id not in self._open
        ]
        if missing and not self.divergences:
            divergences.append(
                Divergence(
                    "tool_call", "end", f"recorded calls not made: {', '.join(missing)}"
                )
            )
        recorded = len(self._prompts)
        if self.model.remaining > 0 and not self.divergences:
            divergences.append(
                Divergence(
                    "model_calls",
                    "end",
                    f"the run made {self._model_calls} model calls,"
                    f" the recording has {recorded}",
                )
            )
        return ReplayReport(
            divergences,
            self._model_calls,
            recorded,
            [(c.name, c.args) for c in self.calls],
            self.recorded_tool_calls,
            list(self.connector.sent),
            notes,
        )
