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

"""`RobotManifest`: the live graph, annotated with what RAI can do there, merged
with the robot spec and rai_whoami; and `ManifestSource`, which rebuilds it when
the graph changes."""

import logging
import os
import threading
from collections.abc import Callable, Iterable
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from rai.tools.hooks import Effect

from .graph import Interfaces, RawGraph, query_graph
from .spec import (
    Capability,
    ManifestSettings,
    RobotSpec,
    WhoamiInfo,
    load_spec,
    load_whoami,
)

if TYPE_CHECKING:
    from rai.communication.ros2 import ROS2Connector

logger = logging.getLogger(__name__)

Kind = Literal["topic", "service", "action"]
KINDS: tuple[Kind, ...] = ("action", "service", "topic")  # rendering order
DEFAULT_MAX_CHARS = 6000

IMAGE_TYPES = frozenset({"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"})
TYPE_FILTERS: dict[str, frozenset[str]] = {"get_ros2_image": IMAGE_TYPES}
"""Generic tools that apply to some types only, by tool name."""

Tools = Iterable[BaseTool] | Callable[[], Iterable[BaseTool]]


class Entry(BaseModel):
    """One topic, service or action of the live graph."""

    name: str
    types: list[str]
    motion: bool = False  # a type that moves hardware (egress MOTION + spec)
    forbidden: bool = False  # the spec says do not use
    access: str = ""  # "r", "w" or "rw" with the generic tools; "" none or unknown
    tools: list[str] = []  # tools configured for this name (their targets)
    capabilities: list[str] = []  # spec capabilities that use it

    @property
    def important(self) -> bool:
        return bool(self.motion or self.forbidden or self.tools or self.capabilities)

    def line(self) -> str:
        flags = ["MOTION"] * self.motion + ["FORBIDDEN"] * self.forbidden
        flags += [self.access] if self.access else []
        flags += [f"cap:{c}" for c in self.capabilities]
        flags += [f"tool:{t}" for t in self.tools]
        line = f"{self.name} {','.join(self.types)}"
        return f"{line} [{', '.join(flags)}]" if flags else line


class CapabilityStatus(Capability):
    """A spec capability with resolved names; `missing` are not in the graph."""

    missing: list[str] = []


class RobotManifest(BaseModel):
    """What the robot is and what its running graph offers.

    `render()` is the compact text for a prompt, `model_dump_json()` the JSON
    form. Both are deterministic for the same inputs.
    """

    name: str | None = None
    description: str | None = None
    whoami: WhoamiInfo | None = None
    frames: dict[str, str] = {}
    limits: dict[str, float | int | str | bool] = {}
    safety: list[str] = []
    capabilities: list[CapabilityStatus] = []
    generic_tools: dict[str, list[str]] = {}  # kind -> tools taking any such name
    nodes: list[str] = []
    topics: list[Entry] = []
    services: list[Entry] = []
    actions: list[Entry] = []
    fingerprint: str = ""  # of the filtered graph; changes when the graph does

    def entries(self, kind: Kind) -> list[Entry]:
        return {"topic": self.topics, "service": self.services, "action": self.actions}[
            kind
        ]

    def render(self, max_chars: int = DEFAULT_MAX_CHARS) -> str:
        """The manifest as text of at most `max_chars` (about 4 chars per token).

        The robot description, safety, limits, frames and capabilities come
        first. Graph entries that matter (MOTION, forbidden, used by a
        capability or a configured tool) are kept before the rest; whatever
        does not fit is counted in an "omitted" line.
        """
        head = self._head(max_chars)
        sections: list[tuple[str, list[str]]] = [
            (title, [e.line() for e in self.entries(kind)])
            for kind, title in zip(KINDS, ("Actions", "Services", "Topics"))
        ] + [("Nodes", list(self.nodes))]
        legend = self._legend()
        candidates = sorted(
            (
                (0 if i < 3 and self.entries(KINDS[i])[j].important else 1, i, j)
                for i, (_, lines) in enumerate(sections)
                for j in range(len(lines))
            )
        )
        reserve = 160  # the "omitted" line
        budget = max_chars - sum(len(line) + 1 for line in head + legend) - reserve
        budget -= sum(len(t) + 12 for t, lines in sections if lines)  # headers
        kept: set[tuple[int, int]] = set()
        for _, i, j in candidates:
            cost = len(sections[i][1][j]) + 1
            if cost <= budget:
                kept.add((i, j))
                budget -= cost
        out = head + legend
        omitted: list[str] = []
        for i, (title, lines) in enumerate(sections):
            if not lines:
                continue
            shown = [line for j, line in enumerate(lines) if (i, j) in kept]
            count = (
                f"{len(lines)}"
                if len(shown) == len(lines)
                else (f"{len(shown)} of {len(lines)}")
            )
            out += [f"### {title} ({count})", *shown]
            if len(shown) < len(lines):
                omitted.append(f"{len(lines) - len(shown)} {title.lower()}")
        if omitted:
            out.append(
                f"Omitted for length: {', '.join(omitted)}. Call get_robot_manifest"
                " with format=json for the full list."
            )
        text = "\n".join(out)
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

    def _head(self, max_chars: int) -> list[str]:
        out = [f"# Robot manifest: {self.name or 'unnamed robot'}"]
        if self.description:
            out.append(_clip(self.description, max_chars // 6))
        if self.whoami:
            who = self.whoami
            parts = [f"Self-description (rai_whoami): {who.description}"] * bool(
                who.description
            )
            for label, items in (
                ("Rules", who.rules),
                ("Capabilities", who.capabilities),
                ("Behaviors", who.behaviors),
            ):
                if items:
                    parts.append(f"{label}: " + "; ".join(items))
            out.append(_clip("\n".join(parts), max_chars // 4))
        if self.safety:
            out += ["## Safety", *(f"- {s}" for s in self.safety)]
        if self.limits:
            out += [
                "## Limits",
                *(f"- {k}: {_value(v)}" for k, v in self.limits.items()),
            ]
        if self.frames:
            out += ["## Frames", *(f"- {k}: {v}" for k, v in self.frames.items())]
        if self.capabilities:
            out.append("## Capabilities")
            out += [_capability_line(c, self) for c in self.capabilities]
        return out

    def _legend(self) -> list[str]:
        if not (self.topics or self.services or self.actions or self.nodes):
            return ["## ROS 2 graph", "(nothing discovered yet)"]
        out = ["## ROS 2 graph"]
        entries = self.topics + self.services + self.actions
        flags = []
        if any(e.motion for e in entries):
            flags.append("MOTION moves hardware")
        if any(e.forbidden for e in entries):
            flags.append("FORBIDDEN do not use")
        if any(e.access for e in entries):
            flags.append("r/w readable/writable with the generic tools")
        if flags:
            out.append("Flags: " + "; ".join(flags) + ".")
        if self.generic_tools:
            out.append(
                "Generic tools: "
                + "; ".join(
                    f"{kind}s: {', '.join(names)}"
                    for kind, names in self.generic_tools.items()
                )
            )
        return out


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(limit - 3, 0)] + "..."


def _value(value: Any) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def _capability_line(cap: CapabilityStatus, manifest: RobotManifest) -> str:
    line = f"- {cap.name}" + (f": {cap.description}" if cap.description else "")
    parts = []
    for kind, names in (
        ("topics", cap.topics),
        ("services", cap.services),
        ("actions", cap.actions),
    ):
        if names:
            shown = [f"{n} (missing)" if n in cap.missing else n for n in names]
            parts.append(f"{kind} {', '.join(shown)}")
    if parts:
        line += " Uses " + "; ".join(parts) + "."
    if cap.examples:
        line += " Examples: " + "; ".join(cap.examples)
    return line


def _tool_kinds(tool: BaseTool) -> set[str]:
    """The kinds of names a generic tool takes from the LLM (its target_args)."""
    return set(getattr(tool, "target_args", {}).values())


def _effect(tool: BaseTool) -> Effect:
    return getattr(tool, "effect", Effect.ACTUATE)


def _permits(tool: Any, kind: str, name: str) -> bool:
    try:
        if kind == "topic" and _effect(tool) is Effect.READ:
            return tool.is_readable(name) if hasattr(tool, "is_readable") else True
        if hasattr(tool, "is_writable"):
            return tool.is_writable(name, service=kind == "service")
        return True
    except Exception:  # an unresolvable name: no access
        return False


def build_manifest(
    connector: "ROS2Connector",
    spec: RobotSpec | None = None,
    whoami: WhoamiInfo | None = None,
    tools: Iterable[BaseTool] = (),
    *,
    include: Iterable[str] = (),
    hide: Iterable[str] = (),
    nodes: bool = True,
    motion_types: Iterable[str] | None = None,
    graph: RawGraph | None = None,
) -> RobotManifest:
    """Query the graph (unless `graph` is given) and merge it with the rest.

    `motion_types` defaults to `rai.communication.ros2.egress.MOTION`; the
    spec's `motion_types` are added to it.
    """
    spec = spec or RobotSpec()
    tools = list(tools)
    if motion_types is None:
        from rai.communication.ros2.egress import MOTION

        motion_types = MOTION
    motion = frozenset(motion_types) | frozenset(spec.motion_types)
    if graph is None:
        graph = query_graph(
            connector, include=include, hide=[*spec.hide, *hide], nodes=nodes
        )

    capabilities = [
        CapabilityStatus(
            **cap.model_dump(exclude={"topics", "services", "actions"}),
            topics=[connector.resolve(n, "topic") for n in cap.topics],
            services=[connector.resolve(n, "service") for n in cap.services],
            actions=[connector.resolve(n, "action") for n in cap.actions],
        )
        for cap in spec.capabilities
    ]
    generic = {
        kind: [t.name for t in tools if kind in _tool_kinds(t)] for kind in KINDS
    }

    def entries(kind: Kind, interfaces: Interfaces) -> list[Entry]:
        out = []
        for name, types in interfaces:
            access = set()
            for tool in tools:
                if kind not in _tool_kinds(tool) or not _permits(tool, kind, name):
                    continue
                if tool.name in TYPE_FILTERS and not TYPE_FILTERS[tool.name] & set(
                    types
                ):
                    continue
                access.add("r" if _effect(tool) is Effect.READ else "w")
            out.append(
                Entry(
                    name=name,
                    types=list(types),
                    motion=bool(motion & set(types)),
                    forbidden=any(fnmatchcase(name, p) for p in spec.forbidden),
                    access="".join(sorted(access)),
                    tools=sorted(
                        {t.name for t in tools if name in getattr(t, "targets", ())}
                    ),
                    capabilities=[
                        c.name for c in capabilities if name in getattr(c, kind + "s")
                    ],
                )
            )
        return out

    manifest = RobotManifest(
        name=spec.name,
        description=spec.description,
        whoami=whoami,
        frames=spec.frames,
        limits=spec.limits,
        safety=spec.safety,
        capabilities=capabilities,
        generic_tools={k: v for k, v in generic.items() if v},
        nodes=list(graph.nodes),
        topics=entries("topic", graph.topics),
        services=entries("service", graph.services),
        actions=entries("action", graph.actions),
        fingerprint=graph.fingerprint(),
    )
    present = {kind: {e.name for e in manifest.entries(kind)} for kind in KINDS}
    for cap in manifest.capabilities:
        cap.missing = [
            n
            for kind in KINDS
            for n in getattr(cap, kind + "s")
            if n not in present[kind]
        ]
    return manifest


class ManifestSource:
    """Builds the manifest on demand and tells when the graph changed.

    No background polling: `changed()` re-reads the graph (a local cache
    read) and compares fingerprints; `current()` rebuilds only when it changed.
    `tools` may be a callable, read at every build (e.g. the tools of an MCP
    server, which can grow after setup).
    """

    def __init__(
        self,
        connector: "ROS2Connector",
        *,
        spec: RobotSpec | str | os.PathLike[str] | None = None,
        whoami: Any = None,
        tools: Tools = (),
        include: Iterable[str] = (),
        hide: Iterable[str] = (),
        nodes: bool = True,
        motion_types: Iterable[str] | None = None,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self.connector = connector
        self.spec = spec if isinstance(spec, RobotSpec | None) else load_spec(spec)
        self.whoami = load_whoami(whoami)
        self.tools = tools
        self.include = tuple(include)
        self.hide = (*(self.spec.hide if self.spec else ()), *hide)
        self.nodes = nodes
        self.motion_types = motion_types
        self.max_chars = max_chars
        self._manifest: RobotManifest | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_settings(
        cls,
        connector: "ROS2Connector",
        settings: ManifestSettings | None = None,
        *,
        tools: Tools = (),
    ) -> "ManifestSource":
        """A source configured by the `[manifest]` table of `rai.toml`."""
        from .spec import manifest_settings

        s = settings or manifest_settings()
        return cls(
            connector,
            spec=s.spec,
            whoami=s.whoami,
            tools=tools,
            include=s.include,
            hide=s.hide,
            nodes=s.nodes,
            max_chars=s.max_chars,
        )

    def graph(self) -> RawGraph:
        return query_graph(
            self.connector, include=self.include, hide=self.hide, nodes=self.nodes
        )

    def build(self) -> RobotManifest:
        """Rebuild now."""
        return self._build(self.graph())

    def _build(self, graph: RawGraph) -> RobotManifest:
        tools = self.tools() if callable(self.tools) else self.tools
        manifest = build_manifest(
            self.connector,
            self.spec,
            self.whoami,
            tools,
            motion_types=self.motion_types,
            graph=graph,
        )
        with self._lock:
            self._manifest = manifest
        return manifest

    def changed(self) -> bool:
        """Whether the graph differs from the last build (True before any)."""
        last = self._manifest
        return last is None or self.graph().fingerprint() != last.fingerprint

    def refresh(self) -> RobotManifest | None:
        """The new manifest if the graph changed since the last build, else None."""
        graph = self.graph()
        last = self._manifest
        if last is not None and graph.fingerprint() == last.fingerprint:
            return None
        return self._build(graph)

    def current(self) -> RobotManifest:
        """The manifest of the graph as it is now."""
        manifest = self.refresh()
        if manifest is not None:
            return manifest
        assert self._manifest is not None
        return self._manifest

    def render(self) -> str:
        """`current().render(max_chars)`."""
        return self.current().render(self.max_chars)
