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

import random
import threading
import time
import uuid
from typing import Any, Generator, List, Optional, Tuple

import numpy as np
import pytest
import rclpy
from cv_bridge import CvBridge
from nav2_msgs.action import NavigateToPose
from pydub import AudioSegment
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import CallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import SetBool
from tf2_ros import TransformBroadcaster, TransformStamped

from rai_interfaces.msg import HRIMessage


class HRIMessagePublisher(Node):
    def __init__(self, topic: str):
        super().__init__("test_hri_message_publisher")
        self.publisher = self.create_publisher(HRIMessage, topic, 10)
        self.timer = self.create_timer(0.1, self.publish_message)
        self.cv_bridge = CvBridge()

    def publish_message(self) -> None:
        msg = HRIMessage()
        image = self.cv_bridge.cv2_to_imgmsg(np.zeros((100, 100, 3), dtype=np.uint8))
        msg.images = [image]
        msg.audios = [AudioSegment.silent(duration=1000)]
        msg.text = "Hello, HRI!"
        self.publisher.publish(msg)


class HRIMessageSubscriber(Node):
    def __init__(self, topic: str):
        super().__init__("test_hri_message_subscriber")
        self.subscription = self.create_subscription(
            HRIMessage, topic, self.handle_test_message, 10
        )
        self.received_messages: List[HRIMessage] = []

    def handle_test_message(self, msg: HRIMessage) -> None:
        self.received_messages.append(msg)


class ServiceServer(Node):
    def __init__(
        self, service_name: str, callback_group: Optional[CallbackGroup] = None
    ):
        super().__init__("test_service_server")
        self.srv = self.create_service(
            SetBool,
            service_name,
            self.handle_test_service,
            callback_group=callback_group,
        )

    def handle_test_service(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        time.sleep(random.random())
        response.success = True
        response.message = "Test service called"
        return response


class ImagePublisher(Node):
    def __init__(self, topic: str):
        super().__init__("test_image_publisher")
        self.publisher = self.create_publisher(Image, topic, 10)  # type: ignore
        self.timer = self.create_timer(0.1, self.publish_image)  # type: ignore

    def publish_image(self) -> None:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()  # type: ignore
        msg.header.frame_id = "test_frame"  # type: ignore
        msg.height = 100
        msg.width = 100
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 300
        msg.data = np.zeros((100, 100, 3), dtype=np.uint8).tobytes()  # type: ignore
        self.publisher.publish(msg)


class MessageSubscriber(Node):
    def __init__(self, topic: str, msg_type: Any = String):
        super().__init__("test_message_subscriber")
        self.msg_type = msg_type
        self.subscription = self.create_subscription(
            msg_type, topic, self.handle_test_message, 10
        )
        self.received_messages: List[msg_type] = []

    def handle_test_message(self, msg: Any) -> None:
        self.received_messages.append(msg)


class TestActionServer(Node):
    __test__ = False

    def __init__(self, action_name: str):
        super().__init__(f"test_action_server_{str(uuid.uuid4())[-12:]}")
        self.action_server = ActionServer(
            self,
            action_type=NavigateToPose,
            action_name=action_name,
            execute_callback=self.handle_test_action,
            goal_callback=self.goal_accepted,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.cancelled: bool = False
        self._shutdown_requested = False
        self._active_actions = 0
        self._actions_lock = threading.Lock()
        self._all_actions_complete = threading.Event()
        self._all_actions_complete.set()  # Initially set (no actions running)

    def handle_test_action(
        self, goal_handle: ServerGoalHandle
    ) -> NavigateToPose.Result:
        with self._actions_lock:
            self._active_actions += 1
            self._all_actions_complete.clear()

        try:
            for i in range(1, 11):
                if self._shutdown_requested:
                    # Node is being destroyed, abort gracefully
                    result = NavigateToPose.Result()
                    result.error_code = 3
                    return result
                if goal_handle.is_cancel_requested:
                    print("Cancel detected in execute callback")
                    try:
                        goal_handle.canceled()
                    except Exception:
                        # Node may be destroyed, return result anyway
                        pass
                    result = NavigateToPose.Result()
                    result.error_code = 3
                    return result
                try:
                    feedback_msg = NavigateToPose.Feedback(distance_remaining=10.0 / i)
                    goal_handle.publish_feedback(feedback_msg)
                except Exception:
                    # Publisher may be invalid if node is being destroyed
                    # Return result to exit gracefully
                    result = NavigateToPose.Result()
                    result.error_code = 3
                    return result
                time.sleep(0.01)

            try:
                goal_handle.succeed()
            except Exception:
                # Node may be destroyed, return result anyway
                pass

            result = NavigateToPose.Result()
            result.error_code = NavigateToPose.Result.NONE
            return result
        finally:
            with self._actions_lock:
                self._active_actions -= 1
                if self._active_actions == 0:
                    self._all_actions_complete.set()

    def goal_accepted(self, goal_handle: ServerGoalHandle) -> GoalResponse:
        self.get_logger().info("Got goal, accepting")
        return GoalResponse.ACCEPT

    def cancel_callback(self, cancel_request) -> CancelResponse:
        self.get_logger().info("Got cancel request")
        self.cancelled = True
        return CancelResponse.ACCEPT


class TestActionClient(Node):
    __test__ = False

    def __init__(self):
        super().__init__("navigate_to_pose_client")
        self._action_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def send_goal(self):
        goal_msg = NavigateToPose.Goal()

        self.get_logger().info("Waiting for action server...")
        self._action_client.wait_for_server()
        self.get_logger().info("Sending goal")

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        self.get_logger().info("Goal sent")

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.get_logger().info("Goal accepted")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: {result}")

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f"Received feedback: {feedback}")


class TestServiceClient(Node):
    __test__ = False

    def __init__(self):
        super().__init__("set_bool_client")
        self.client = self.create_client(SetBool, "set_bool")

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service...")

        self.req = SetBool.Request()
        self.req.data = True

    def send_request(self):
        self.future = self.client.call_async(self.req)


class MessagePublisher(Node):
    def __init__(self, topic: str):
        super().__init__(f"test_message_publisher_{str(uuid.uuid4())[-12:]}")
        self.publisher = self.create_publisher(String, topic, 10)
        self.timer = self.create_timer(0.1, self.publish_message)

    def publish_message(self) -> None:
        msg = String()
        msg.data = "Hello, ROS2!"
        self.publisher.publish(msg)


class TransformPublisher(Node):
    def __init__(self, topic: str, use_sim_time: bool = False):
        super().__init__("test_transform_publisher")
        if use_sim_time:
            self.set_parameters(
                [
                    rclpy.parameter.Parameter(
                        "use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True
                    )
                ]
            )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.publish_transform)
        self.frame_id = "base_link"
        self.child_frame_id = "map"

    def publish_transform(self) -> None:
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()  # type: ignore
        msg.header.frame_id = self.frame_id  # type: ignore
        msg.child_frame_id = self.child_frame_id  # type: ignore
        msg.transform.translation.x = 1.0  # type: ignore
        msg.transform.translation.y = 2.0  # type: ignore
        msg.transform.translation.z = 3.0  # type: ignore
        msg.transform.rotation.x = 0.0  # type: ignore
        msg.transform.rotation.y = 0.0  # type: ignore
        msg.transform.rotation.z = 0.0  # type: ignore
        msg.transform.rotation.w = 1.0  # type: ignore
        self.tf_broadcaster.sendTransform(msg)


class ClockPublisher(Node):
    def __init__(self):
        super().__init__("test_clock_publisher")
        self.publisher = self.create_publisher(Clock, "/clock", 10)
        self.timer = self.create_timer(0.1, self.publish_clock)
        self.start_time = time.time()

    def publish_clock(self) -> None:
        msg = Clock()
        # Publish simulation time that advances faster than real time
        sim_time = (time.time() - self.start_time) * 2.0  # 2x speed
        msg.clock.sec = int(sim_time)
        msg.clock.nanosec = int((sim_time % 1.0) * 1e9)
        self.publisher.publish(msg)


def multi_threaded_spinner(
    nodes: List[Node],
) -> Tuple[List[MultiThreadedExecutor], List[threading.Thread]]:
    executors: List[MultiThreadedExecutor] = []
    executor_threads: List[threading.Thread] = []
    for node in nodes:
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executors.append(executor)
    for executor in executors:
        executor_thread = threading.Thread(target=executor.spin)
        executor_thread.daemon = True
        executor_thread.start()
        executor_threads.append(executor_thread)
    return executors, executor_threads


def shutdown_executors_and_threads(
    executors: List[MultiThreadedExecutor], threads: List[threading.Thread]
) -> None:
    # Collect all nodes first (before shutting down executors)
    all_nodes = []
    for executor in executors:
        try:
            all_nodes.extend(executor.get_nodes())
        except Exception:
            pass

    # Signal action servers to stop executing
    action_servers = []
    for node in all_nodes:
        try:
            if isinstance(node, TestActionServer):
                node._shutdown_requested = True
                action_servers.append(node)
        except Exception:
            pass

    # Wait for actions to complete gracefully (with timeout as fallback)
    if action_servers:
        for server in action_servers:
            server._all_actions_complete.wait(timeout=0.15)

    # Cancel all timers BEFORE shutting down executors
    # This prevents executors from trying to call timers after they're canceled
    for node in all_nodes:
        try:
            # Try to access and cancel timers through node's internal structure
            if hasattr(node, "_timers"):
                timers = node._timers
                if isinstance(timers, dict):
                    for timer in list(timers.values()):
                        try:
                            timer.cancel()
                        except Exception:
                            pass
        except Exception:
            pass

    # Give executor a moment to process timer cancellations and remove them from wait list
    time.sleep(0.1)

    # Now shutdown executors to stop spinning threads
    for executor in executors:
        try:
            executor.shutdown()
        except Exception as e:
            print(f"Error shutting down executor: {e}")

    # Wait for threads to actually finish (shutdown() is async)
    for thread in threads:
        thread.join(timeout=2.0)

    # Clean up any remaining nodes (after executors are shut down and threads joined)
    for node in all_nodes:
        try:
            node.destroy_node()
        except Exception as e:
            print(f"Error destroying node: {e}")


def create_mock_clock_with_time(use_time_wrapper: bool = False):
    """Create a mock clock that returns a compatible Time message.

    This utility handles ROS2 Humble vs Jazzy differences:
    - Humble: Strict type checking in __debug__ mode requires actual BuiltinTime
      instances, not MagicMock objects. Using MagicMock causes AssertionError.
    - Jazzy: More lenient with MagicMock, but BuiltinTime instances don't allow
      dynamically adding methods (AttributeError when accessing to_msg).

    Parameters
    ----------
    use_time_wrapper : bool, optional
        If True, creates a TimeWithToMsg wrapper that adds to_msg() method
        for compatibility when time.to_msg() is called multiple times.
        If False, returns a simple Time message (default).

    Returns
    -------
    tuple
        (mock_clock, mock_time) where:
        - mock_clock: MagicMock configured with now().to_msg() returning mock_time
        - mock_time: Either a Time message or TimeWithToMsg instance
    """
    from unittest.mock import MagicMock

    from builtin_interfaces.msg import Time as BuiltinTime

    if use_time_wrapper:

        class TimeWithToMsg(BuiltinTime):
            """BuiltinTime wrapper that adds to_msg() method for compatibility."""

            def to_msg(self):
                return self

        mock_time = TimeWithToMsg()
    else:
        mock_time = BuiltinTime(sec=1234567890, nanosec=0)

    mock_clock = MagicMock()
    mock_clock.now.return_value.to_msg.return_value = mock_time

    return mock_clock, mock_time


def setup_mock_clock_for_node(node, use_time_wrapper: bool = False):
    """Setup mock clock for a node.

    Parameters
    ----------
    node : Any
        Node object (or mock) that has a get_clock() method
    use_time_wrapper : bool, optional
        See create_mock_clock_with_time() for details

    Returns
    -------
    tuple
        (mock_clock, mock_time) from create_mock_clock_with_time()
    """
    from unittest.mock import MagicMock

    mock_clock, mock_time = create_mock_clock_with_time(use_time_wrapper)
    node.get_clock = MagicMock(return_value=mock_clock)
    return mock_clock, mock_time


def setup_mock_clock_for_agent(agent, use_time_wrapper: bool = True):
    """Setup mock clock for an agent with ros2_connector.

    This is a convenience wrapper for agents that have agent.ros2_connector._node.

    Parameters
    ----------
    agent : Any
        Agent object with ros2_connector._node attribute
    use_time_wrapper : bool, optional
        Defaults to True for agent tests that may call to_msg() multiple times

    Returns
    -------
    tuple
        (mock_clock, mock_time) from create_mock_clock_with_time()
    """
    mock_clock, mock_time = setup_mock_clock_for_node(
        agent.ros2_connector._node, use_time_wrapper=use_time_wrapper
    )
    return mock_clock, mock_time


@pytest.fixture(scope="function")
def ros_setup() -> Generator[None, None, None]:
    rclpy.init()
    yield
    rclpy.shutdown()
