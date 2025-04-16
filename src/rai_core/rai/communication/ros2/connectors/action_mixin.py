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

from typing import Any, Callable, Optional

from rai.communication.ros2.api import ROS2ActionAPI
from rai.communication.ros2.messages import ROS2HRIMessage, ROS2Message


class ROS2ActionMixin:
    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        if not hasattr(self, "_actions_api"):
            raise AttributeError(
                f"{self.__class__.__name__} instance must have an attribute 'actions_api' of type ROS2ActionAPI"
            )
        self._actions_api: ROS2ActionAPI  # to make the type checker happy
        if not isinstance(self._actions_api, ROS2ActionAPI):
            raise AttributeError(
                f"{self.__class__.__name__} instance must have an attribute 'actions_api' of type ROS2ActionAPI"
            )

    def start_action(
        self,
        action_data: Optional[ROS2Message | ROS2HRIMessage],
        target: str,
        on_feedback: Callable[[Any], None] = lambda _: None,
        on_done: Callable[[Any], None] = lambda _: None,
        timeout_sec: float = 1.0,
        *,
        msg_type: str,
        **kwargs: Any,
    ) -> str:
        if not isinstance(action_data, ROS2Message):
            raise ValueError("Action data must be of type ROS2Message")
        accepted, handle = self._actions_api.send_goal(
            action_name=target,
            action_type=msg_type,
            goal=action_data.payload,
            timeout_sec=timeout_sec,
            feedback_callback=on_feedback,
            done_callback=on_done,
        )
        if not accepted:
            raise RuntimeError("Action goal was not accepted")
        return handle

    def terminate_action(self, action_handle: str, **kwargs: Any):
        return self._actions_api.terminate_goal(action_handle)
