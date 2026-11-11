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

"""FlightRecorder: one MCAP file per session (see :mod:`rai.recorder`)."""

import atexit
import base64
import json
import logging
import os
import queue
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Self, Union

from rai.agents.base import AgentState, BaseAgent
from rai.agents.events import (
    AgentError,
    AgentEvent,
    FinalMessage,
    Interrupted,
    ModelEnd,
    ModelStart,
    Observation,
    StateChanged,
    ToolEnd,
    ToolStart,
)

from . import schemas as s
from .encode import prepare, ref_of

if TYPE_CHECKING:
    from rai.safety import AuditLog, AuditRecord

logger = logging.getLogger(__name__)

_STOP = object()
_DEBUG, _INFO, _WARNING, _ERROR = 1, 2, 3, 4  # foxglove.Log levels


def _require_mcap() -> Any:
    try:
        import mcap.writer
    except ImportError as e:
        raise ImportError(
            "rai.recorder needs the 'recorder' extra: pip install 'rai_core[recorder]'"
        ) from e
    return mcap.writer


class _Clock:
    """Stamps in ns on one clock: ROS time when the connector's node runs on
    sim time, else the wall clock (read through the monotonic clock, so it
    never steps backwards during a session)."""

    def __init__(self, connector: Any, mode: str) -> None:
        self._offset = time.time_ns() - time.monotonic_ns()
        self._ros: Any = None
        self.source = "wall"
        if mode == "auto" and connector is not None:
            clock = connector._runtime.node.get_clock()
            if getattr(clock, "ros_time_is_active", False):
                self._ros, self.source = clock, "ros"

    def now(self) -> int:
        if self._ros is not None:
            try:
                return self._ros.now().nanoseconds
            except Exception:  # the node is gone: keep stamping
                pass
        return time.monotonic_ns() + self._offset

    def at(self, monotonic_ns: int) -> int:
        """The stamp of an instant taken with time.monotonic_ns()."""
        if self._ros is None:
            return monotonic_ns + self._offset
        return self.now() - (time.monotonic_ns() - monotonic_ns)


def _time(ns: int) -> dict[str, int]:
    return {"sec": ns // 1_000_000_000, "nsec": ns % 1_000_000_000}


def _short(value: Any, limit: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=repr)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _text(message: Any) -> str:
    return str(getattr(message, "text", ""))  # a property in LangChain 1.x


class FlightRecorder:
    """Records an agent session into one MCAP file.

    Channels (JSON with JSON schemas unless noted):

    - ``/rai/agent/events``: every agent event, one compact line each.
    - ``/rai/agent/llm``: model requests (model name, the prompt as LangChain
      message dicts, sent as a delta against the agent's previous request) and
      responses (message, token usage, latency).
    - ``/rai/agent/tools``: tool call start (args, effect) and end (status,
      content, artifact, duration).
    - ``/rai/agent/observations``: state a StateBasedAgent added to a request;
      its images on ``/rai/agent/observations/<source>/image``.
    - ``/rai/agent/images``: other images (prompts, tool artifacts),
      ``foxglove.CompressedImage``. Every image is stored once; JSON refers to
      it by ``rai-image:<ref>`` (``frame_id`` is the ref). WAV audio goes to
      attachments named ``<ref>.wav``.
    - ``/rai/agent/log``: a readable timeline, ``foxglove.Log``.
    - ``/rai/safety/decisions``: every hook decision of both hook points
      (AuditLog records).
    - each recorded ROS topic under its own name, raw CDR with its ``ros2msg``
      schema, as rosbag2 writes it.

    ``log_time`` and ``publish_time`` are equal: when it happened, on one clock
    (ROS time if the connector runs on sim time, else the wall clock).
    Recording never blocks the agent or the executor: inputs only stamp and
    enqueue (``maxsize``), one writer thread encodes and writes, and a full
    queue drops (``dropped``). The file is flushed every ``flush_s``, so a
    crash loses at most that much.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        agent: Union[BaseAgent, Iterable[BaseAgent], None] = None,
        topics: Union[Iterable[str], Mapping[str, Any]] = (),
        audit: Union["AuditLog", bool] = True,
        connector: Any = None,
        clock: Literal["auto", "wall"] = "auto",
        maxsize: int = 10_000,
        flush_s: float = 1.0,
        chunk_size: int = 1 << 20,
        compression: Literal["zstd", "lz4", "none"] = "zstd",
    ) -> None:
        """
        Args:
            path: the .mcap file (parents are created; an existing file is
                replaced).
            agent: agents to record; more with :meth:`attach`.
            topics: ROS topics recorded raw: names (type from the ROS graph,
                waited for if the topic does not exist yet) or ``{name: type}``.
            audit: an AuditLog to take hook decisions from; True installs a
                private one on every runtime and RaiTool; False: none.
            connector: the ROS2Connector for topics and the clock; one on the
                shared runtime is created if topics need it.
            clock: "auto" uses ROS time when the connector is on sim time.
        """
        self._mcap = _require_mcap()
        self.path = Path(path)
        self.maxsize = maxsize
        self.flush_s = flush_s
        self.chunk_size = chunk_size
        self.compression = compression
        self.dropped = 0
        self.dropped_by_input: Counter[str] = Counter()
        self.recorded = 0
        self.errors = 0
        self._clock_mode = clock
        self._agents: list[BaseAgent] = []
        self._unsubscribe: list[Callable[[], None]] = []
        if isinstance(agent, BaseAgent):
            self._agents.append(agent)
        elif agent is not None:
            self._agents += list(agent)
        if isinstance(topics, Mapping):
            self._topics: dict[str, Any] = dict(topics)
        else:
            self._topics = {name: None for name in topics}
        self._audit_arg = audit
        self._audit: Optional["AuditLog"] = None
        self._connector = connector
        self._owned: list[Any] = []
        self._subscriptions: list[Any] = []
        self._queue: queue.Queue[Any] = queue.Queue(maxsize)
        self._lock = threading.Lock()
        self._state: Literal["created", "running", "stopped"] = "created"
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._discovery: Optional[threading.Thread] = None
        self._clock: Optional[_Clock] = None
        # writer-thread state
        self._writer: Any = None
        self._file: Any = None
        self._schemas: dict[str, int] = {}
        self._channels: dict[str, int] = {}
        self._sequence: Counter[int] = Counter()
        self._media: set[str] = set()
        self._media_channel = s.IMAGES
        self._media_stamp = 0
        self._prompts: dict[str, tuple[Any, ...]] = {}
        self._model_t0: dict[str, float] = {}
        self._tool_t0: dict[str, float] = {}

    # ------------------------------------------------------------ lifecycle

    def start(self) -> Self:
        """Open the file and start recording. Raises if the file cannot be
        opened or the connector cannot be created."""
        with self._lock:
            if self._state != "created":
                raise RuntimeError("a FlightRecorder can only be started once")
            self._state = "running"
        try:
            self._open()
            self._attach_inputs()
        except BaseException:
            self.stop()
            raise
        atexit.register(self._atexit)
        return self

    def _open(self) -> None:
        if self._topics and self._connector is None:
            from rai.communication.ros2 import ROS2Connector

            self._connector = ROS2Connector()
            self._owned.append(self._connector)
        self._clock = _Clock(self._connector, self._clock_mode)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("wb")
        compression = {
            "zstd": self._mcap.CompressionType.ZSTD,
            "lz4": self._mcap.CompressionType.LZ4,
            "none": self._mcap.CompressionType.NONE,
        }[self.compression]
        self._writer = self._mcap.Writer(
            self._file, chunk_size=self.chunk_size, compression=compression
        )
        self._writer.start(profile="ros2", library="rai.recorder")
        self._writer.add_metadata("rai.session", self._session())
        self._writer.flush()
        self._thread = threading.Thread(
            target=self._run, name="rai_recorder", daemon=True
        )
        self._thread.start()

    def _session(self) -> dict[str, str]:
        assert self._clock is not None
        try:
            from importlib.metadata import version

            rai_version = version("rai_core")
        except Exception:
            rai_version = ""
        return {
            "format": "rai.recorder",
            "format_version": s.FORMAT_VERSION,
            "clock": self._clock.source,
            "start_ns": str(self._clock.now()),
            "start_wall_ns": str(time.time_ns()),
            "agents": json.dumps([a.name for a in self._agents]),
            "topics": json.dumps(list(self._topics)),
            "ros_distro": os.environ.get("ROS_DISTRO", ""),
            "rai_version": rai_version,
        }

    def _attach_inputs(self) -> None:
        for agent in self._agents:
            self._unsubscribe.append(agent.subscribe(self._on_event))
        if self._audit_arg is True:
            from rai.safety import AuditLog

            self._audit = AuditLog().install()
            self._owned.insert(0, self._audit)
        elif self._audit_arg:
            self._audit = self._audit_arg
        if self._audit is not None:
            self._unsubscribe.append(self._audit.subscribe(self._on_audit))
        pending = {n: t for n, t in self._topics.items() if not self._record(n, t)}
        if pending:
            self._discovery = threading.Thread(
                target=self._discover,
                args=(pending,),
                name="rai_recorder_discovery",
                daemon=True,
            )
            self._discovery.start()

    def attach(self, agent: BaseAgent) -> Self:
        """Record another agent (before or after start)."""
        self._agents.append(agent)
        if self._state == "running":
            self._unsubscribe.append(agent.subscribe(self._on_event))
        return self

    def stop(self, timeout: float = 10.0) -> None:
        """Stop recording and finish the file. Idempotent."""
        with self._lock:
            if self._state == "stopped":
                return
            started, self._state = self._state == "running", "stopped"
        self._stopped.set()
        atexit.unregister(self._atexit)
        for unsubscribe in self._unsubscribe:
            unsubscribe()
        self._unsubscribe.clear()
        for owned in [o for o in self._owned if o is self._audit]:
            owned.close()  # its queued decisions reach us first
            self._owned.remove(owned)
        for subscription in self._subscriptions:
            subscription.close()
        self._subscriptions.clear()
        if self._discovery is not None:
            self._discovery.join(timeout)
        if started and self._thread is not None:
            try:
                self._queue.put(_STOP, timeout=timeout)
            except queue.Full:
                logger.error("FlightRecorder writer is stuck; %s may be cut", self.path)
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger.error("FlightRecorder writer did not finish %s", self.path)
        elif self._file is not None:
            self._file.close()
        for owned in self._owned:
            owned.close()
        self._owned.clear()

    uninstall = stop

    def flush(self, timeout: float = 10.0) -> None:
        """Wait until everything recorded so far is in the file."""
        if self._audit is not None:
            self._audit.flush()
        if self._state != "running" or self._thread is None:
            return
        done = threading.Event()
        self._queue.put(done, timeout=timeout)
        done.wait(timeout)

    def _atexit(self) -> None:
        self.stop(timeout=2.0)

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------ inputs

    def _put(self, source: str, item: tuple[Any, ...]) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self.dropped += 1
                self.dropped_by_input[source] += 1
                first = self.dropped == 1
            if first:
                logger.warning("FlightRecorder queue full: dropping (%s)", source)

    def _on_event(self, event: AgentEvent) -> None:  # the agent's thread
        assert self._clock is not None
        self._put("agent", ("event", self._clock.now(), event))

    def _on_audit(self, record: "AuditRecord") -> None:  # the AuditLog thread
        assert self._clock is not None
        self._put("audit", ("audit", self._clock.at(record.monotonic_ns), record))

    def _record(self, topic: str, msg_type: Any) -> bool:
        """Subscribe to a topic raw; False if its type is not known yet."""
        from rai.communication.ros2.egress import type_name
        from rai.communication.ros2.runtime import message_class

        connector = self._connector
        try:
            name = connector.resolve(topic)
            if msg_type is None:
                types = dict(connector.get_topics_names_and_types()).get(name)
                if not types:
                    return False
                msg_type = types[0]
            ros_type = type_name(message_class(msg_type))
            subscription = connector.subscribe(
                name,
                lambda data: self._on_raw(name, ros_type, data),
                msg_type=msg_type,
                raw=True,
                maxsize=None,
            )
        except Exception as e:
            logger.debug("FlightRecorder: %s not recordable yet: %r", topic, e)
            return False
        self._subscriptions.append(subscription)
        return True

    def _discover(self, pending: dict[str, Any]) -> None:
        warned = time.monotonic() + 5.0
        while pending and not self._stopped.wait(0.2):
            for topic, msg_type in list(pending.items()):
                if self._stopped.is_set():
                    return
                if self._record(topic, msg_type):
                    del pending[topic]
            if pending and time.monotonic() > warned:
                logger.warning(
                    "FlightRecorder: waiting for %s to appear", ", ".join(pending)
                )
                warned = float("inf")

    def _on_raw(self, topic: str, type_name: str, data: bytes) -> None:
        assert self._clock is not None
        self._put(topic, ("raw", self._clock.now(), (topic, type_name, data)))

    # ------------------------------------------------------------ writer thread

    def _run(self) -> None:
        last_flush = time.monotonic()
        while True:
            try:
                item = self._queue.get(timeout=self.flush_s)
            except queue.Empty:
                item = None
            if item is _STOP:
                break
            if isinstance(item, threading.Event):
                self._safe(self._writer.flush)
                item.set()
                last_flush = time.monotonic()
                continue
            if item is not None:
                self._safe(self._write, item)
            if time.monotonic() - last_flush >= self.flush_s:
                self._safe(self._writer.flush)
                last_flush = time.monotonic()
        self._finish()

    def _safe(self, fn: Callable[..., Any], *args: Any) -> None:
        try:
            fn(*args)
        except Exception:
            self.errors += 1
            if self.errors == 1:
                logger.exception("FlightRecorder failed to write to %s", self.path)

    def _finish(self) -> None:
        try:
            self._writer.add_metadata(
                "rai.recorder.stats",
                {
                    "recorded": str(self.recorded),
                    "dropped": str(self.dropped),
                    "dropped_by_input": json.dumps(dict(self.dropped_by_input)),
                    "errors": str(self.errors),
                    "end_ns": str(self._clock.now() if self._clock else 0),
                },
            )
            self._writer.finish()
        except Exception:
            logger.exception("FlightRecorder could not finish %s", self.path)
        finally:
            self._file.close()

    def _write(self, item: tuple[Any, ...]) -> None:
        kind, stamp, data = item
        if kind == "event":
            self._agent_event(stamp, data)
        elif kind == "audit":
            self._decision(stamp, data)
        elif kind == "raw":
            self._raw(stamp, *data)

    def _channel(self, topic: str, schema: str) -> int:
        channel = self._channels.get(topic)
        if channel is None:
            schema_id = self._schemas.get(schema)
            if schema_id is None:
                schema_id = self._writer.register_schema(
                    schema, "jsonschema", json.dumps(s.SCHEMAS[schema]).encode()
                )
                self._schemas[schema] = schema_id
            channel = self._writer.register_channel(topic, "json", schema_id)
            self._channels[topic] = channel
        return channel

    def _add(self, channel: int, stamp: int, data: bytes) -> None:
        self._sequence[channel] += 1
        self._writer.add_message(
            channel,
            log_time=stamp,
            publish_time=stamp,
            data=data,
            sequence=self._sequence[channel],
        )
        self.recorded += 1

    def _json(self, topic: str, stamp: int, value: Any, schema: str = "") -> None:
        channel = self._channel(topic, schema or s.CHANNEL_SCHEMAS[topic])
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        self._add(channel, stamp, data.encode())

    def _prepare(self, value: Any, stamp: int, media_channel: str = s.IMAGES) -> Any:
        self._media_channel, self._media_stamp = media_channel, stamp
        return prepare(value, self._store_media)

    def _store_media(self, kind: str, fmt: str, data: str) -> str:
        ref = ref_of(data)
        if ref in self._media:
            return ref
        self._media.add(ref)
        stamp = self._media_stamp
        if kind == "image":
            image = {
                "timestamp": _time(stamp),
                "frame_id": ref,
                "data": data,
                "format": fmt,
            }
            self._json(self._media_channel, stamp, image, "foxglove.CompressedImage")
        else:
            self._writer.add_attachment(
                stamp, stamp, f"{ref}.{fmt}", f"{kind}/{fmt}", base64.b64decode(data)
            )
        return ref

    def _log(self, stamp: int, level: int, name: str, message: str) -> None:
        self._json(
            s.LOG,
            stamp,
            {
                "timestamp": _time(stamp),
                "level": level,
                "message": message,
                "name": name,
                "file": "",
                "line": 0,
            },
        )

    def _agent_event(self, stamp: int, e: AgentEvent) -> None:
        turn = getattr(e, "turn", "")
        summary: dict[str, Any] = {
            "agent": e.agent,
            "turn": turn,
            "event": type(e).__name__,
            "t": e.t,
        }
        common = {"agent": e.agent, "turn": turn}
        if isinstance(e, StateChanged):
            summary |= {"old": str(e.old), "new": str(e.new)}
            level = _ERROR if e.new is AgentState.FAILED else _DEBUG
            self._log(stamp, level, e.agent, f"state {e.old} -> {e.new}")
        elif isinstance(e, ModelStart):
            summary["model"] = e.model
            self._model_t0[e.agent] = e.t
            prefix, new = self._delta(e.agent, e.messages)
            request = common | {
                "phase": "request",
                "model": e.model,
                "prefix": prefix,
                "messages": self._prepare(new, stamp),
            }
            self._json(s.LLM, stamp, request)
            self._log(
                stamp, _DEBUG, e.agent, f"LLM request to {e.model or 'model'}"
                f" ({len(e.messages)} messages, {len(new)} new)",
            )  # fmt: skip
        elif isinstance(e, ModelEnd):
            message = e.message
            meta = message.response_metadata or {}
            t0 = self._model_t0.pop(e.agent, None)
            response = common | {
                "phase": "response",
                "model": meta.get("model_name") or meta.get("model") or "",
                "message": self._prepare(message, stamp),
                "text": _text(message),
                "tool_calls": self._prepare(message.tool_calls, stamp),
                "usage": dict(message.usage_metadata or {}),
                "latency_ms": None if t0 is None else (e.t - t0) * 1000,
            }
            self._json(s.LLM, stamp, response)
            calls = ", ".join(c["name"] for c in message.tool_calls)
            text = _short(_text(message), 500) or "(no text)"
            self._log(
                stamp,
                _INFO,
                e.agent,
                f"LLM: {text}" + (f" -> {calls}" if calls else ""),
            )
        elif isinstance(e, Observation):
            values = {
                name: self._prepare(value, stamp, s.observation_images(name))
                for name, value in e.values.items()
            }
            observation = common | {"fresh": list(e.fresh), "values": values}
            self._json(s.OBSERVATIONS, stamp, observation)
            self._log(
                stamp,
                _DEBUG,
                e.agent,
                f"observation ({', '.join(e.fresh) or 'none'} new)",
            )
        elif isinstance(e, ToolStart):
            summary |= {"call_id": e.call_id, "name": e.name}
            self._tool_t0[e.call_id] = e.t
            start = common | {
                "phase": "start",
                "call_id": e.call_id,
                "name": e.name,
                "effect": str(e.effect),
                "args": self._prepare(e.args, stamp),
            }
            self._json(s.TOOLS, stamp, start)
            self._log(stamp, _INFO, e.agent, f"tool {e.name}({_short(e.args)})")
        elif isinstance(e, ToolEnd):
            summary |= {"call_id": e.call_id, "name": e.name, "status": e.status}
            t0 = self._tool_t0.pop(e.call_id, None)
            end = common | {
                "phase": "end",
                "call_id": e.call_id,
                "name": e.name,
                "effect": str(e.effect),
                "status": e.status,
                "content": self._prepare(e.content, stamp),
                "artifact": self._prepare(e.artifact, stamp),
                "duration_ms": None if t0 is None else (e.t - t0) * 1000,
            }
            self._json(s.TOOLS, stamp, end)
            level = {"success": _INFO, "denied": _WARNING}.get(e.status, _ERROR)
            content = _short(end["content"])
            self._log(stamp, level, e.agent, f"tool {e.name} {e.status}: {content}")
        elif isinstance(e, FinalMessage):
            self._log(stamp, _INFO, e.agent, f"final: {_short(_text(e.message), 500)}")
        elif isinstance(e, Interrupted):
            self._log(stamp, _WARNING, e.agent, "turn interrupted")
        elif isinstance(e, AgentError):
            summary["error"] = f"{type(e.error).__name__}: {e.error}"
            self._log(stamp, _ERROR, e.agent, f"turn failed: {summary['error']}")
        self._json(s.EVENTS, stamp, summary)

    def _delta(self, agent: str, messages: tuple[Any, ...]) -> tuple[int, list[Any]]:
        """How many leading messages the request shares with the agent's
        previous one, and the rest."""
        previous = self._prompts.get(agent, ())
        prefix = 0
        for old, new in zip(previous, messages):
            if old is not new and old != new:
                break
            prefix += 1
        self._prompts[agent] = messages
        return prefix, list(messages[prefix:])

    def _decision(self, stamp: int, record: "AuditRecord") -> None:
        value = self._prepare(record.to_dict(), stamp)
        self._json(s.DECISIONS, stamp, value)
        if record.decision == "deny":
            self._log(
                stamp,
                _WARNING,
                "safety",
                f"{record.hook} denied {record.kind} {record.name}: {record.reason}",
            )

    def _raw(self, stamp: int, topic: str, type_name: str, data: bytes) -> None:
        channel = self._channels.get(topic)
        if channel is None:
            schema_id = self._schemas.get(type_name)
            if schema_id is None:
                try:
                    encoding, text = s.ros2_schema(type_name)
                except Exception:
                    logger.warning("No schema for %s; recording it without", type_name)
                    encoding, text = "", ""
                schema_id = self._writer.register_schema(
                    type_name, encoding, text.encode()
                )
                self._schemas[type_name] = schema_id
            channel = self._writer.register_channel(topic, "cdr", schema_id)
            self._channels[topic] = channel
        self._add(channel, stamp, bytes(data))


def install(
    path: Union[str, Path], *agents: BaseAgent, **kwargs: Any
) -> FlightRecorder:
    """Start a :class:`FlightRecorder` of agents into path (keywords as for
    FlightRecorder); ``stop()`` or ``uninstall()`` finishes the file."""
    return FlightRecorder(path, agent=agents, **kwargs).start()
