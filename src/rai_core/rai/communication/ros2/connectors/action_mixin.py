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

from typing import Any, Callable, Optional, Union

from rai.communication.ros2.api import IROS2Message, ROS2ActionAPI
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
        action_data: Optional[Union[ROS2Message, ROS2HRIMessage, IROS2Message]],
        target: str,
        on_feedback: Callable[[Any], None] = lambda _: None,
        on_done: Callable[[Any], None] = lambda _: None,
        timeout_sec: float = 1.0,
        *,
        msg_type: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Start a ROS2 action.

        Provides dual support:
        - LLM support: ROS2Message with dict payload + msg_type string
        - Typed (human-friendly): Direct action Goal class instance (msg_type inferred)

        Parameters
        ----------
        action_data : Optional[Union[ROS2Message, ROS2HRIMessage, IROS2Message]]
            The action goal. Can be:
            - ROS2Message/ROS2HRIMessage with dict payload (requires msg_type)
            - Action Goal class instance (e.g., MoveGroup.Goal(), msg_type optional)
        target : str
            The target action name.
        on_feedback : Callable[[Any], None], optional
            Callback for feedback messages, by default lambda _: None.
        on_done : Callable[[Any], None], optional
            Callback when action completes, by default lambda _: None.
        timeout_sec : float, optional
            Timeout in seconds, by default 1.0.
        msg_type : str | None, optional
            The ROS2 action type string (e.g., 'moveit_msgs/action/MoveGroup').
            Required if action_data is ROS2Message (dict), optional if action_data is Goal instance.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        str
            Action handle for tracking the action.

        Raises
        ------
        RuntimeError
            If action goal was not accepted.
        ValueError
            If action_data is invalid or msg_type is missing for dict-based messages.
        """
        # Hybrid support: handle both ROS2Message (dict) and Goal class instances
        if isinstance(action_data, (ROS2Message, ROS2HRIMessage)):
            # LLM support path: dict-based message
            if msg_type is None:
                raise ValueError(
                    "msg_type must be provided when action_data is ROS2Message (dict-based). "
                    "Either pass msg_type or use a Goal class instance directly."
                )
            goal = action_data.payload
        elif action_data is None:
            raise ValueError("action_data cannot be None")
        else:
            # Typed (human-friendly) path: Goal class instance
            goal = action_data

        accepted, handle = self._actions_api.send_goal(
            action_name=target,
            action_type=msg_type,
            goal=goal,
            timeout_sec=timeout_sec,
            feedback_callback=on_feedback,
            done_callback=on_done,
        )
        if not accepted:
            raise RuntimeError("Action goal was not accepted")
        return handle

    def terminate_action(self, action_handle: str, **kwargs: Any):
        return self._actions_api.terminate_goal(action_handle)
