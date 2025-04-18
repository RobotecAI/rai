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

import uuid
from collections import defaultdict
from functools import partial
from threading import Lock
from typing import Any, Callable, Dict, List, Type

from langchain_core.tools import BaseTool  # type: ignore
from langchain_core.utils import stringify_dict
from pydantic import BaseModel, Field
from rclpy.action import CancelResponse

from rai.communication.ros2 import ROS2Message
from rai.tools.ros2.base import BaseROS2Tool, BaseROS2Toolkit

internal_action_id_mapping: Dict[str, str] = {}
action_results_store: Dict[str, Any] = {}
action_results_store_lock: Any = Lock()
action_feedbacks_store: Dict[str, List[Any]] = defaultdict(list)
action_feedbacks_store_lock: Any = Lock()


def get_internal_action_id_mapping():
    return internal_action_id_mapping


def get_action_results_store():
    return action_results_store


def get_action_results_store_lock():
    return action_results_store_lock


def get_action_feedbacks_store():
    return action_feedbacks_store


def get_action_feedbacks_store_lock():
    return action_feedbacks_store_lock


class ROS2ActionToolkit(BaseROS2Toolkit):
    name: str = "ros2_action"
    description: str = "A toolkit for ROS2 actions"

    action_results_store: Dict[str, Any] = Field(
        default_factory=get_action_results_store
    )
    action_results_store_lock: Any = Field(
        default_factory=get_action_results_store_lock
    )
    action_feedbacks_store: Dict[str, List[Any]] = Field(
        default_factory=get_action_feedbacks_store
    )
    action_feedbacks_store_lock: Any = Field(
        default_factory=get_action_feedbacks_store_lock
    )

    def get_tools(self) -> List[BaseTool]:
        return [
            StartROS2ActionTool(
                connector=self.connector,
                feedback_callback=self._generic_feedback_callback,
                on_done_callback=self._generic_on_done_callback,
                readable=self.readable,
                writable=self.writable,
                forbidden=self.forbidden,
            ),
            CancelROS2ActionTool(
                connector=self.connector,
                readable=self.readable,
                writable=self.writable,
                forbidden=self.forbidden,
            ),
            GetROS2ActionFeedbackTool(
                connector=self.connector,
                readable=self.readable,
                writable=self.writable,
                forbidden=self.forbidden,
            ),
            GetROS2ActionResultTool(),
            GetROS2ActionIDsTool(),
            GetROS2ActionsNamesAndTypesTool(
                connector=self.connector,
                readable=self.readable,
                writable=self.writable,
                forbidden=self.forbidden,
            ),
        ]

    def _generic_feedback_callback(self, action_id: str, feedback: Any) -> None:
        with self.action_feedbacks_store_lock:
            self.action_feedbacks_store[action_id].append(feedback)

    def _generic_on_done_callback(self, action_id: str, future: Any) -> None:
        with self.action_results_store_lock:
            self.action_results_store[action_id] = future.result().result


class GetROS2ActionsNamesAndTypesToolInput(BaseModel):
    pass


class GetROS2ActionsNamesAndTypesTool(BaseROS2Tool):
    name: str = "get_ros2_actions_names_and_types"
    description: str = "Get the names and types of all ROS2 actions"
    args_schema: Type[GetROS2ActionsNamesAndTypesToolInput] = (
        GetROS2ActionsNamesAndTypesToolInput
    )

    def _run(self) -> str:
        actions_and_types = self.connector.get_actions_names_and_types()
        if all([self.readable is None, self.writable is None, self.forbidden is None]):
            response = [
                {"action": action, "type": type} for action, type in actions_and_types
            ]
            return "\n".join([stringify_dict(action) for action in response])
        else:
            writable_actions: List[Dict[str, Any]] = []

            for action, type in actions_and_types:
                if self.is_writable(action):
                    writable_actions.append({"action": action, "type": type})
                    continue

            text_response = "\n".join(
                [
                    stringify_dict(action_description)
                    for action_description in writable_actions
                ]
            )
            return text_response


class StartROS2ActionToolInput(BaseModel):
    action_name: str = Field(..., description="The name of the action to start")
    action_type: str = Field(..., description="The type of the action")
    action_args: Dict[str, Any] = Field(
        ..., description="The arguments to pass to the action"
    )


class StartROS2ActionTool(BaseROS2Tool):
    feedback_callback: Callable[[Any, str], None] = lambda _, __: None
    on_done_callback: Callable[[Any, str], None] = lambda _, __: None
    internal_action_id_mapping: Dict[str, str] = Field(
        default_factory=get_internal_action_id_mapping
    )
    name: str = "start_ros2_action"
    description: str = "Start a ROS2 action"
    args_schema: Type[StartROS2ActionToolInput] = StartROS2ActionToolInput

    def _run(
        self, action_name: str, action_type: str, action_args: Dict[str, Any]
    ) -> str:
        if not self.is_writable(action_name):
            raise ValueError(f"Action {action_name} is not writable")
        message = ROS2Message(payload=action_args)
        action_id = str(uuid.uuid4())
        response = self.connector.start_action(
            message,
            action_name,
            on_feedback=partial(self.feedback_callback, action_id),
            on_done=partial(self.on_done_callback, action_id),
            msg_type=action_type,
        )
        self.internal_action_id_mapping[response] = action_id
        return "Action started with ID: " + response


class GetROS2ActionFeedbackToolInput(BaseModel):
    action_id: str = Field(..., description="The ID of the action to get feedback for")


class GetROS2ActionFeedbackTool(BaseROS2Tool):
    name: str = "get_ros2_action_feedback"
    description: str = "Get the feedback of a ROS2 action by its action ID"
    args_schema: Type[GetROS2ActionFeedbackToolInput] = GetROS2ActionFeedbackToolInput

    action_feedbacks_store: Dict[str, List[Any]] = Field(
        default_factory=get_action_feedbacks_store
    )
    action_feedbacks_store_lock: Any = Field(
        default_factory=get_action_feedbacks_store_lock
    )
    internal_action_id_mapping: Dict[str, str] = Field(
        default_factory=get_internal_action_id_mapping
    )

    def _run(self, action_id: str) -> str:
        with self.action_feedbacks_store_lock:
            external_action_id = self.internal_action_id_mapping[action_id]
            feedbacks = self.action_feedbacks_store[external_action_id]
            self.action_feedbacks_store[external_action_id] = []
            return str(feedbacks)


class GetROS2ActionResultToolInput(BaseModel):
    action_id: str = Field(
        ..., description="The id of the action to get the result for"
    )


class GetROS2ActionResultTool(BaseTool):
    name: str = "get_ros2_action_result"
    description: str = "Get the result of a ROS2 action by its id"
    args_schema: Type[GetROS2ActionResultToolInput] = GetROS2ActionResultToolInput

    action_results_store: Dict[str, Any] = Field(
        default_factory=get_action_results_store
    )
    action_results_store_lock: Any = Field(
        default_factory=get_action_results_store_lock
    )
    internal_action_id_mapping: Dict[str, str] = Field(
        default_factory=get_internal_action_id_mapping
    )

    def _run(self, action_id: str) -> str:
        with self.action_results_store_lock:
            external_action_id = self.internal_action_id_mapping[action_id]
            result = self.action_results_store[external_action_id]
            return str(result)


class CancelROS2ActionToolInput(BaseModel):
    action_id: str = Field(..., description="The ID of the action to cancel")


class CancelROS2ActionTool(BaseROS2Tool):
    name: str = "cancel_ros2_action"
    description: str = "Cancel a ROS2 action"
    args_schema: Type[CancelROS2ActionToolInput] = CancelROS2ActionToolInput

    def _run(self, action_id: str) -> str:
        response = self.connector.terminate_action(action_id)
        if response == CancelResponse.ACCEPT:
            return f"Action {action_id} cancelled."
        else:
            return f"Action {action_id} could not be cancelled."


class GetROS2ActionIDsToolInput(BaseModel):
    pass


class GetROS2ActionIDsTool(BaseTool):
    name: str = "get_ros2_action_ids"
    description: str = "Get the IDs of all ROS2 actions"
    args_schema: Type[GetROS2ActionIDsToolInput] = GetROS2ActionIDsToolInput
    internal_action_id_mapping: Dict[str, str] = Field(
        default_factory=get_internal_action_id_mapping
    )

    def _run(self) -> str:
        return str(list(self.internal_action_id_mapping.keys()))
