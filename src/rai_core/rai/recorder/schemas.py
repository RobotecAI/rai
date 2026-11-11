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

"""Channel names, JSON schemas and ROS 2 message definitions of a recording."""

import functools
import re
from pathlib import Path
from typing import Any

FORMAT_VERSION = "1"

EVENTS = "/rai/agent/events"
LLM = "/rai/agent/llm"
TOOLS = "/rai/agent/tools"
OBSERVATIONS = "/rai/agent/observations"
IMAGES = "/rai/agent/images"
LOG = "/rai/agent/log"
DECISIONS = "/rai/safety/decisions"


def observation_images(source: str) -> str:
    """The image channel of one state source."""
    return f"{OBSERVATIONS}/{re.sub(r'[^A-Za-z0-9_]', '_', source)}/image"


_TIME = {
    "type": "object",
    "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
}
_ANY: dict[str, Any] = {}
_COMMON = {
    "agent": {"type": "string"},
    "turn": {"type": "string", "description": "id of the agent turn (ainvoke)"},
}


def _schema(title: str, properties: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"title": title, "type": "object", "properties": properties, **extra}


SCHEMAS: dict[str, dict[str, Any]] = {
    "rai.AgentEvent": _schema(
        "rai.AgentEvent",
        {
            **_COMMON,
            "event": {"type": "string", "description": "the event class name"},
            "t": {"type": "number", "description": "wall clock, s since epoch"},
            "old": {"type": "string"},
            "new": {"type": "string"},
            "model": {"type": "string"},
            "call_id": {"type": "string"},
            "name": {"type": "string"},
            "status": {"type": "string"},
            "error": {"type": "string"},
        },
    ),
    "rai.LLMCall": _schema(
        "rai.LLMCall",
        {
            **_COMMON,
            "phase": {"enum": ["request", "response"]},
            "model": {"type": "string"},
            "prefix": {
                "type": "integer",
                "description": "request: the prompt is the first `prefix` messages"
                " of this agent's previous request, then `messages`",
            },
            "messages": {"type": "array", "items": _ANY},
            "message": _ANY,
            "text": {"type": "string"},
            "tool_calls": {"type": "array", "items": _ANY},
            "usage": {
                "type": "object",
                "properties": {
                    "input_tokens": {"type": "integer"},
                    "output_tokens": {"type": "integer"},
                    "total_tokens": {"type": "integer"},
                },
            },
            "latency_ms": {"type": "number"},
        },
    ),
    "rai.ToolCall": _schema(
        "rai.ToolCall",
        {
            **_COMMON,
            "phase": {"enum": ["start", "end"]},
            "call_id": {"type": "string"},
            "name": {"type": "string"},
            "effect": {"type": "string"},
            "args": {"type": "object"},
            "status": {"enum": ["success", "error", "denied"]},
            "content": _ANY,
            "artifact": _ANY,
            "duration_ms": {"type": "number"},
        },
    ),
    "rai.Observation": _schema(
        "rai.Observation",
        {
            **_COMMON,
            "fresh": {"type": "array", "items": {"type": "string"}},
            "values": {"type": "object"},
        },
    ),
    "rai.HookDecision": _schema(
        "rai.HookDecision",
        {
            "time": {"type": "number", "description": "wall clock, s since epoch"},
            "side": {"enum": ["tool", "egress"]},
            "origin": {"type": ["string", "null"]},
            "kind": {"type": "string"},
            "name": {"type": "string"},
            "type": {"type": "string"},
            "decision": {"enum": ["allow", "deny"]},
            "hook": {"type": "string"},
            "reason": {"type": "string"},
            "lease_s": {"type": "number"},
            "leased": {"type": "boolean"},
            "targets": {"type": "array", "items": {"type": "string"}},
            "args": {"type": "object"},
        },
    ),
    "foxglove.CompressedImage": _schema(
        "foxglove.CompressedImage",
        {
            "timestamp": _TIME,
            "frame_id": {"type": "string"},
            "data": {"type": "string", "contentEncoding": "base64"},
            "format": {"type": "string"},
        },
    ),
    "foxglove.Log": _schema(
        "foxglove.Log",
        {
            "timestamp": _TIME,
            "level": {"type": "integer", "enum": [0, 1, 2, 3, 4, 5]},
            "message": {"type": "string"},
            "name": {"type": "string"},
            "file": {"type": "string"},
            "line": {"type": "integer"},
        },
    ),
}

CHANNEL_SCHEMAS = {
    EVENTS: "rai.AgentEvent",
    LLM: "rai.LLMCall",
    TOOLS: "rai.ToolCall",
    OBSERVATIONS: "rai.Observation",
    IMAGES: "foxglove.CompressedImage",
    LOG: "foxglove.Log",
    DECISIONS: "rai.HookDecision",
}


# ------------------------------------------------------------ ROS 2 types

_PRIMITIVES = frozenset(
    "bool byte char float32 float64 int8 uint8 int16 uint16 int32 uint32"
    " int64 uint64 string wstring".split()
)
_SEPARATOR = "=" * 80


def _share(package: str) -> Path:
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory(package))


def _msg_file(package: str, name: str) -> Path:
    path = _share(package) / "msg" / f"{name}.msg"
    if not path.is_file():
        raise LookupError(f"no .msg file for {package}/msg/{name}")
    return path


def _dependencies(package: str, text: str) -> list[str]:
    """The message types ("pkg/Name") the fields of a .msg text refer to."""
    deps: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        # constants and defaults only exist on primitive fields
        base = re.sub(r"\[.*\]$", "", line.split()[0])
        base = re.sub(r"<=\d+$", "", base)
        if base in _PRIMITIVES:
            continue
        parts = base.split("/")
        if len(parts) == 1:
            dep = f"{package}/{parts[0]}"
        else:
            dep = f"{parts[0]}/{parts[-1]}"
        if dep not in deps:
            deps.append(dep)
    return deps


@functools.cache
def ros2_msgdef(type_name: str) -> str:
    """The full ``ros2msg`` definition of a message type ("pkg/msg/Name"),
    with its dependencies, as rosbag2 writes it to MCAP."""
    package, _, name = type_name.split("/")
    main = _msg_file(package, name).read_text()
    seen = {f"{package}/{name}"}
    pending = _dependencies(package, main)
    parts = [main.strip()]
    while pending:
        dep = pending.pop(0)
        if dep in seen:
            continue
        seen.add(dep)
        dep_package, dep_name = dep.split("/")
        text = _msg_file(dep_package, dep_name).read_text()
        parts.append(f"{_SEPARATOR}\nMSG: {dep}\n{text.strip()}")
        pending += _dependencies(dep_package, text)
    return "\n".join(parts) + "\n"


def ros2_schema(type_name: str) -> tuple[str, str]:
    """(encoding, data) of a ROS 2 type's schema: ``ros2msg`` from the .msg
    files, or what rosbag2 finds (e.g. ``ros2idl`` for action messages)."""
    try:
        return "ros2msg", ros2_msgdef(type_name)
    except LookupError:
        pass
    try:
        from rosbag2_py import LocalMessageDefinitionSource  # type: ignore
    except ImportError:
        raise LookupError(f"no .msg file for {type_name}") from None

    definition = LocalMessageDefinitionSource().get_full_text(type_name)
    if not definition.encoded_message_definition:
        raise LookupError(f"no message definition found for {type_name}")
    return definition.encoding, definition.encoded_message_definition
