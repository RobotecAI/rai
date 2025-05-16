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

import copy
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypedDict,
    cast,
)

import rclpy
import rclpy.action
import rclpy.node
import rclpy.task
import rosidl_runtime_py.set_message
from action_msgs.srv import CancelGoal
from rclpy.action import ActionClient, CancelResponse, GoalResponse
from rclpy.action.client import ClientGoalHandle
from rclpy.action.server import (
    ActionServer,
    ServerGoalHandle,
    default_cancel_callback,
    default_goal_callback,
    default_handle_accepted_callback,
)
from rclpy.qos import (
    QoSProfile,
    qos_profile_action_status_default,
    qos_profile_services_default,
)
from rclpy.task import Future

from rai.communication.ros2.api.base import (
    BaseROS2API,
    IROS2Message,
)
from rai.communication.ros2.api.conversion import import_message_from_str


class ROS2ActionData(TypedDict):
    action_client: Optional[ActionClient]
    goal_future: Optional[rclpy.task.Future]
    result_future: Optional[rclpy.task.Future]
    client_goal_handle: Optional[ClientGoalHandle]
    feedbacks: List[Any]


class ROS2ActionAPI(BaseROS2API):
    def __init__(self, node: rclpy.node.Node) -> None:
        self.node = node
        self._logger = node.get_logger()
        self.actions: Dict[str, ROS2ActionData] = {}
        self._action_servers: Dict[str, ActionServer] = {}
        self._callback_executor = ThreadPoolExecutor(max_workers=10)

    def _generate_handle(self):
        return str(uuid.uuid4())

    def _generic_callback(self, handle: str, feedback_msg: Any) -> None:
        self.actions[handle]["feedbacks"].append(feedback_msg.feedback)

    def _fan_out_feedback(
        self, callbacks: List[Callable[[Any], None]], feedback_msg: Any
    ) -> None:
        """Fan out feedback message to multiple callbacks concurrently.

        Args:
            callbacks: List of callback functions to execute
            feedback_msg: The feedback message to pass to each callback
        """
        for callback in callbacks:
            self._callback_executor.submit(
                self._safe_callback_wrapper, callback, feedback_msg
            )

    def _safe_callback_wrapper(
        self, callback: Callable[[Any], None], feedback_msg: Any
    ) -> None:
        """Safely execute a callback with error handling.

        Args:
            callback: The callback function to execute
            feedback_msg: The feedback message to pass to the callback
        """
        try:
            callback(copy.deepcopy(feedback_msg))
        except Exception as e:
            self._logger.error(f"Error in feedback callback: {str(e)}")

    def create_action_server(
        self,
        action_type: str,
        action_name: str,
        execute_callback: Callable[[ServerGoalHandle], Type[IROS2Message]],
        *,
        callback_group: Optional[rclpy.node.CallbackGroup] = None,
        goal_callback: Callable[[IROS2Message], GoalResponse] = default_goal_callback,
        handle_accepted_callback: Callable[
            [ServerGoalHandle], None
        ] = default_handle_accepted_callback,
        cancel_callback: Callable[
            [IROS2Message], CancelResponse
        ] = default_cancel_callback,
        goal_service_qos_profile: QoSProfile = qos_profile_services_default,
        result_service_qos_profile: QoSProfile = qos_profile_services_default,
        cancel_service_qos_profile: QoSProfile = qos_profile_services_default,
        feedback_pub_qos_profile: QoSProfile = QoSProfile(depth=10),
        status_pub_qos_profile: QoSProfile = qos_profile_action_status_default,
        result_timeout: int = 900,
    ) -> str:
        """
        Create an action server.

        Args:
            action_type: The action message type with namespace
            action_name: The name of the action server
            execute_callback: The callback to execute when a goal is received
            callback_grou: The callback group to use for the action server
            goal_callback: The callback to execute when a goal is received
            handle_accepted_callback: The callback to execute when a goal handle is accepted
            cancel_callback: The callback to execute when a goal is canceled
            goal_service_qos_profile: The QoS profile for the goal service
            result_service_qos_profile: The QoS profile for the result service
            cancel_service_qos_profile: The QoS profile for the cancel service
            feedback_pub_qos_profile: The QoS profile for the feedback publisher
            status_pub_qos_profile: The QoS profile for the status publisher
            result_timeout: The timeout for waiting for a result

        Returns:
            The handle for the created action server

        Raises:
            ValueError: If the action server cannot be created
        """
        handle = self._generate_handle()
        action_ros_type = import_message_from_str(action_type)
        try:
            action_server = ActionServer(
                node=self.node,
                action_type=action_ros_type,
                action_name=action_name,
                execute_callback=execute_callback,
                callback_group=callback_group,
                goal_callback=goal_callback,
                handle_accepted_callback=handle_accepted_callback,
                cancel_callback=cancel_callback,
                goal_service_qos_profile=goal_service_qos_profile,
                result_service_qos_profile=result_service_qos_profile,
                cancel_service_qos_profile=cancel_service_qos_profile,
                feedback_pub_qos_profile=feedback_pub_qos_profile,
                status_pub_qos_profile=status_pub_qos_profile,
                result_timeout=result_timeout,
            )
            self._logger.info(f"Created action server: {action_name}")
        except TypeError as e:
            import inspect

            signature = inspect.signature(ActionServer.__init__)
            args = [
                param.name
                for param in signature.parameters.values()
                if param.name != "self"
            ]

            raise ValueError(
                f"Failed to create action server: {str(e)}. Valid arguments are: {args}"
            )
        self._action_servers[handle] = action_server
        return handle

    def send_goal(
        self,
        action_name: str,
        action_type: str,
        goal: Dict[str, Any],
        *,
        feedback_callback: Callable[[Any], None] = lambda _: None,
        done_callback: Callable[
            [Any], None
        ] = lambda _: None,  # TODO: handle done callback
        timeout_sec: float = 1.0,
    ) -> Tuple[bool, Annotated[str, "action handle"]]:
        handle = self._generate_handle()
        self.actions[handle] = ROS2ActionData(
            action_client=None,
            goal_future=None,
            result_future=None,
            client_goal_handle=None,
            feedbacks=[],
        )

        action_cls = import_message_from_str(action_type)
        action_goal = action_cls.Goal()  # type: ignore
        rosidl_runtime_py.set_message.set_message_fields(action_goal, goal)

        action_client = ActionClient(self.node, action_cls, action_name)
        if not action_client.wait_for_server(timeout_sec=timeout_sec):  # type: ignore
            return False, ""

        feedback_callbacks = [
            partial(self._generic_callback, handle),
            feedback_callback,
        ]
        send_goal_future: Future = action_client.send_goal_async(
            goal=action_goal,
            feedback_callback=partial(self._fan_out_feedback, feedback_callbacks),
        )
        self.actions[handle]["action_client"] = action_client
        self.actions[handle]["goal_future"] = send_goal_future

        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            if send_goal_future.done():
                break
            time.sleep(0.01)

        goal_handle = cast(Optional[ClientGoalHandle], send_goal_future.result())
        if goal_handle is None:
            return False, ""

        get_result_future = cast(Future, goal_handle.get_result_async())  # type: ignore
        get_result_future.add_done_callback(done_callback)  # type: ignore

        self.actions[handle]["result_future"] = get_result_future
        self.actions[handle]["client_goal_handle"] = goal_handle

        return goal_handle.accepted, handle  # type: ignore

    def terminate_goal(self, handle: str) -> CancelGoal.Response:
        if self.actions[handle]["client_goal_handle"] is None:
            raise ValueError(
                f"Cannot terminate goal {handle} as it was not accepted or has no goal handle."
            )
        return self.actions[handle]["client_goal_handle"].cancel_goal()

    def get_feedback(self, handle: str) -> List[Any]:
        return self.actions[handle]["feedbacks"]

    def is_goal_done(self, handle: str) -> bool:
        if handle not in self.actions:
            raise ValueError(f"Invalid action handle: {handle}")
        if self.actions[handle]["result_future"] is None:
            raise ValueError(
                f"Result future is None for handle: {handle}. Was the goal accepted?"
            )
        return self.actions[handle]["result_future"].done()

    def get_result(self, handle: str) -> Any:
        if not self.is_goal_done(handle):
            raise ValueError(f"Goal {handle} is not done")
        if self.actions[handle]["result_future"] is None:
            raise ValueError(f"No result available for goal {handle}")
        return self.actions[handle]["result_future"].result()

    def get_action_names_and_types(self) -> List[Tuple[str, List[str]]]:
        return rclpy.action.get_action_names_and_types(self.node)

    def shutdown(self) -> None:
        """Cleanup thread pool when object is destroyed."""
        if hasattr(self, "_callback_executor"):
            self._callback_executor.shutdown(wait=False)
