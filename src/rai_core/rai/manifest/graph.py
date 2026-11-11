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

"""The live half of the manifest: the ROS 2 graph, without the noise.

Graph queries read rmw's local discovery cache, so a query (and the
fingerprint used to detect changes) costs about a millisecond and no traffic.
"""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rai.communication.ros2 import ROS2Connector

Interfaces = tuple[tuple[str, tuple[str, ...]], ...]

NOISE_TOPICS = frozenset({"/rosout", "/parameter_events"})
PARAMETER_SERVICES = frozenset(
    {
        "describe_parameters",
        "get_parameter_types",
        "get_parameters",
        "get_type_description",
        "list_parameters",
        "set_parameters",
        "set_parameters_atomically",
    }
)


def is_noise(kind: str, name: str) -> bool:
    """Hidden names (a segment starting with `_`, which covers action
    internals), rosout and parameter events, and every node's parameter services."""
    if any(part.startswith("_") for part in name.split("/")):
        return True
    if kind == "topic":
        return name in NOISE_TOPICS
    if kind == "service":
        return name.rsplit("/", 1)[-1] in PARAMETER_SERVICES
    return False


@dataclass(frozen=True)
class RawGraph:
    """The filtered graph, sorted. Equal graphs have equal fingerprints."""

    nodes: tuple[str, ...]
    topics: Interfaces
    services: Interfaces
    actions: Interfaces

    def fingerprint(self) -> str:
        data = json.dumps([self.nodes, self.topics, self.services, self.actions])
        return hashlib.sha256(data.encode()).hexdigest()[:16]


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatchcase(name, p) for p in patterns)


def query_graph(
    connector: "ROS2Connector",
    *,
    include: Iterable[str] = (),
    hide: Iterable[str] = (),
    nodes: bool = True,
) -> RawGraph:
    """Nodes, topics, services and actions with their types. A name is kept
    when it is not noise, matches `include` (if given) and does not match `hide`.
    The connector's own node is left out."""
    include, hide = tuple(include), tuple(hide)

    def keep(kind: str, name: str) -> bool:
        return (
            not is_noise(kind, name)
            and (not include or _matches(name, include))
            and not _matches(name, hide)
        )

    def interfaces(kind: str, pairs: Iterable[tuple[str, list[str]]]) -> Interfaces:
        return tuple(sorted((n, tuple(sorted(t))) for n, t in pairs if keep(kind, n)))

    node_names: tuple[str, ...] = ()
    if nodes:
        node: Any = connector._node  # pyright: ignore[reportPrivateUsage]
        own = node.get_fully_qualified_name()
        node_names = tuple(
            sorted(
                fqn
                for name, ns in node.get_node_names_and_namespaces()
                if (fqn := f"{ns.rstrip('/')}/{name}") != own and keep("node", fqn)
            )
        )
    return RawGraph(
        nodes=node_names,
        topics=interfaces("topic", connector.get_topics_names_and_types()),
        services=interfaces("service", connector.get_services_names_and_types()),
        actions=interfaces("action", connector.get_actions_names_and_types()),
    )
