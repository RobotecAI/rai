# Copyright (C) 2024 Robotec.AI
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

try:
    pass
except ImportError:
    raise ImportError(
        "This is a ROS2 feature. Make sure ROS2 is installed and sourced."
    )

from typing import Any, Callable, Dict, Type

from langchain_core.tools import BaseTool, tool  # type: ignore
from pydantic import BaseModel, Field

from rai.communication.ros2.connectors import ROS2ARIConnector, ROS2ARIMessage


class StartActionToolInput(BaseModel):
    action_name: str = Field(..., description="The name of the action to start")
    action_type: str = Field(..., description="The type of the action")
    args: Dict[str, Any] = Field(..., description="The arguments to pass to the action")


class StartActionTool(BaseTool):
    connector: ROS2ARIConnector
    feedback_callback: Callable[[Any], None] = lambda _: None
    on_done_callback: Callable[[Any], None] = lambda _: None
    name: str = "start_action"
    description: str = "Start a ROS2 action"
    args_schema: Type[StartActionToolInput] = StartActionToolInput

    def _run(self, action_name: str, action_type: str, args: Dict[str, Any]) -> str:
        message = ROS2ARIMessage(payload=args, metadata={"msg_type": action_type})
        response = self.connector.start_action(message, action_name)
        return "Action started with ID: " + response


@tool
def get_action_feedback(action_id: str) -> str:
    """Get the feedback of a ROS2 action by its action ID"""
    raise NotImplementedError("Not implemented")
