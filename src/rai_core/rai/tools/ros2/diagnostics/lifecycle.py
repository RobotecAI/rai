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

"""Lifecycle (managed) node states via each node's get_state service."""

import asyncio
from typing import Optional, Type

from lifecycle_msgs.srv import GetState
from pydantic import BaseModel, Field

from rai.tools.hooks import Effect
from rai.tools.ros2.base import BaseROS2Tool
from rai.tools.ros2.diagnostics.common import bounded, node_name

GET_STATE_TYPE = "lifecycle_msgs/srv/GetState"


class GetROS2LifecycleStatesInput(BaseModel):
    node: Optional[str] = Field(
        None, description="One managed node, e.g. /map_server; all of them if empty"
    )
    timeout_sec: float = Field(2.0, gt=0, description="Timeout of each GetState call")


class GetROS2LifecycleStatesTool(BaseROS2Tool):
    default_effect = Effect.READ
    name: str = "get_ros2_lifecycle_states"
    description: str = (
        "List ROS 2 lifecycle (managed) nodes and their current state"
        " (unconfigured, inactive, active, finalized)"
    )
    args_schema: Type[GetROS2LifecycleStatesInput] = GetROS2LifecycleStatesInput

    async def _state(self, service: str, timeout: float) -> str:
        try:
            response = await self.connector.call(
                service, GetState.Request(), srv_type=GetState, timeout=timeout
            )
        except Exception as e:
            return f"no answer ({type(e).__name__})"
        return response.current_state.label

    async def _acall(self, node: Optional[str] = None, timeout_sec: float = 2.0) -> str:
        services = sorted(
            name
            for name, types in self.connector.get_services_names_and_types()
            if GET_STATE_TYPE in types
            and name.endswith("/get_state")
            and not self._forbidden(name, "service")
        )
        if node:
            service = f"{node_name(node).rstrip('/')}/get_state"
            if service not in services:
                raise ValueError(f"{node_name(node)} is not a lifecycle node")
            services = [service]
        if not services:
            return "No lifecycle nodes in the ROS graph"
        states = await asyncio.gather(*(self._state(s, timeout_sec) for s in services))
        lines = [
            f"{s.removesuffix('/get_state')}: {state}"
            for s, state in zip(services, states)
        ]
        return "\n".join([f"{len(lines)} lifecycle nodes:", *bounded(lines)])
