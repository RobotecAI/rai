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

"""The ROS graph without the ros2 CLI: nodes, what they offer, parameters and
installed interfaces."""

import difflib
from collections import Counter
from typing import List, Optional, Type

import rosidl_runtime_py
from pydantic import BaseModel, Field
from rcl_interfaces.srv import DescribeParameters, GetParameters, ListParameters
from rclpy.parameter import parameter_value_to_python

from rai.tools.hooks import Effect
from rai.tools.ros2.base import BaseROS2Tool
from rai.tools.ros2.diagnostics.common import bounded, clip, node_name

# Every rclpy/rclcpp node serves these; listing them is noise.
PARAMETER_SERVICES = (
    "describe_parameters",
    "get_parameter_types",
    "get_parameters",
    "get_type_description",
    "list_parameters",
    "set_parameters",
    "set_parameters_atomically",
)


def _known_node(tool: BaseROS2Tool, node: str) -> str:
    """The fully qualified node name; a ValueError names close matches."""
    node = node_name(node)
    nodes = tool.connector.get_node_names()
    if node not in nodes:
        close = difflib.get_close_matches(node, nodes, n=3, cutoff=0.5)
        hint = f"; did you mean {', '.join(close)}?" if close else ""
        raise ValueError(f"Node {node} is not in the ROS graph{hint}")
    return node


class ListROS2NodesInput(BaseModel):
    include_hidden: bool = Field(
        False, description="Also list hidden nodes (names starting with _)"
    )


class ListROS2NodesTool(BaseROS2Tool):
    default_effect = Effect.READ
    name: str = "list_ros2_nodes"
    description: str = (
        "List the nodes in the ROS 2 graph; flags duplicate names, which break"
        " parameters and services"
    )
    args_schema: Type[ListROS2NodesInput] = ListROS2NodesInput

    async def _acall(self, include_hidden: bool = False) -> str:
        nodes = Counter(self.connector.get_node_names())
        own = self.connector.node_name
        hidden = [n for n in nodes if n.rsplit("/", 1)[-1].startswith("_")]
        lines = []
        for node, count in sorted(nodes.items()):
            if node in hidden and not include_hidden:
                continue
            notes = [f"x{count}: duplicate name"] if count > 1 else []
            notes += ["this agent"] if node == own else []
            lines.append(f"{node} ({', '.join(notes)})" if notes else node)
        header = f"{len(lines)} nodes"
        if hidden and not include_hidden:
            header += f" ({len(hidden)} hidden not shown)"
        return "\n".join([header + ":", *bounded(lines)])


class GetROS2NodeInfoInput(BaseModel):
    node: str = Field(..., min_length=1, description="Node name, e.g. /amcl")


class GetROS2NodeInfoTool(BaseROS2Tool):
    default_effect = Effect.READ
    name: str = "get_ros2_node_info"
    description: str = (
        "What a ROS 2 node publishes, subscribes to, serves and calls (topics,"
        " services, actions with their types)"
    )
    args_schema: Type[GetROS2NodeInfoInput] = GetROS2NodeInfoInput

    async def _acall(self, node: str) -> str:
        node = _known_node(self, node)
        lines = [f"{node}:"]
        for kind, entries in self.connector.get_node_interfaces(node).items():
            service = kind in ("services", "clients")
            shown, parameter_services = [], 0
            for name, types in sorted(entries):
                if "/_action/" in name or self._forbidden(
                    name, "service" if service else "topic"
                ):
                    continue
                if kind == "services" and name.rsplit("/", 1)[-1] in PARAMETER_SERVICES:
                    parameter_services += 1
                    continue
                shown.append(f"  {name} [{', '.join(types)}]")
            count = len(shown) + parameter_services
            if parameter_services:
                shown.append(f"  ({parameter_services} parameter services)")
            if shown:
                lines += [f"{kind.replace('_', ' ')} ({count}):", *bounded(shown, 50)]
        return "\n".join(lines)


class GetROS2ParametersInput(BaseModel):
    node: str = Field(..., min_length=1, description="Node name, e.g. /amcl")
    names: Optional[List[str]] = Field(
        None, description="Parameter names; all of the node's parameters if empty"
    )
    describe: bool = Field(
        False, description="Add each parameter's description and read-only flag"
    )
    timeout_sec: float = Field(2.0, gt=0, description="Timeout of each service call")


class GetROS2ParametersTool(BaseROS2Tool):
    default_effect = Effect.READ
    name: str = "get_ros2_parameters"
    description: str = "Get the parameter values of a ROS 2 node"
    args_schema: Type[GetROS2ParametersInput] = GetROS2ParametersInput

    async def _call(self, node: str, service: str, request: object, timeout: float):
        name = f"{node}/{service}"
        if self._forbidden(name, "service"):
            raise ValueError(f"Service {name} is forbidden")
        return await self.connector.call(name, request, timeout=timeout)

    async def _acall(
        self,
        node: str,
        names: Optional[List[str]] = None,
        describe: bool = False,
        timeout_sec: float = 2.0,
    ) -> str:
        node = _known_node(self, node)
        if not names:
            listed = await self._call(
                node, "list_parameters", ListParameters.Request(), timeout_sec
            )
            names = sorted(listed.result.names)
        values = await self._call(
            node, "get_parameters", GetParameters.Request(names=names), timeout_sec
        )
        descriptors = None
        if describe:
            described = await self._call(
                node,
                "describe_parameters",
                DescribeParameters.Request(names=names),
                timeout_sec,
            )
            descriptors = described.descriptors
        lines = []
        for i, (name, value) in enumerate(zip(names, values.values)):
            text = f"{name} = {clip(repr(parameter_value_to_python(value)), 120)}"
            if descriptors is not None:
                descriptor = descriptors[i]
                if descriptor.read_only:
                    text += " (read-only)"
                if descriptor.description:
                    text += f"  # {clip(descriptor.description, 120)}"
            lines.append(text)
        return "\n".join([f"{node}: {len(lines)} parameters", *bounded(lines, 200)])


class ListROS2InterfacesInput(BaseModel):
    package: Optional[str] = Field(
        None,
        description="List this package's msg/srv/action types; without it, list"
        " the packages that define interfaces",
    )


class ListROS2InterfacesTool(BaseROS2Tool):
    default_effect = Effect.READ
    name: str = "list_ros2_interfaces"
    description: str = (
        "List installed ROS 2 interface packages, or the message, service and"
        " action types of one package (use get_ros2_message_interface for fields)"
    )
    args_schema: Type[ListROS2InterfacesInput] = ListROS2InterfacesInput

    async def _acall(self, package: Optional[str] = None) -> str:
        if package:
            types = rosidl_runtime_py.get_interfaces([package])[package]
            return "\n".join(f"{package}/{t}" for t in sorted(types))
        lines = []
        for name, types in sorted(rosidl_runtime_py.get_interfaces().items()):
            kinds = Counter(t.split("/", 1)[0] for t in types)
            lines.append(f"{name}: " + ", ".join(f"{n} {k}" for k, n in kinds.items()))
        return "\n".join(bounded(lines, 300))
