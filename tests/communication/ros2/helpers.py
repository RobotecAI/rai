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

import threading
import time
from typing import Any, Generator, List, Tuple

import numpy as np
import pytest
import rclpy
from cv_bridge import CvBridge
from nav2_msgs.action import NavigateToPose
from pydub import AudioSegment
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
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
    def __init__(self, service_name: str):
        super().__init__("test_service_server")
        self.srv = self.create_service(SetBool, service_name, self.handle_test_service)

    def handle_test_service(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
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
    def __init__(self, action_name: str):
        super().__init__("test_action_server")
        self.action_server = ActionServer(
            self,
            action_type=NavigateToPose,
            action_name=action_name,
            execute_callback=self.handle_test_action,
            goal_callback=self.goal_accepted,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )

    def handle_test_action(
        self, goal_handle: ServerGoalHandle
    ) -> NavigateToPose.Result:
        for i in range(1, 11):
            if goal_handle.is_cancel_requested:
                print("Cancel detected in execute callback")
                goal_handle.canceled()
                result = NavigateToPose.Result()
                result.error_code = 3
                return result
            feedback_msg = NavigateToPose.Feedback(distance_remaining=10.0 / i)
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(0.01)

        goal_handle.succeed()

        result = NavigateToPose.Result()
        result.error_code = NavigateToPose.Result.NONE
        return result

    def goal_accepted(self, goal_handle: ServerGoalHandle) -> GoalResponse:
        self.get_logger().info("Got goal, accepting")
        return GoalResponse.ACCEPT

    def cancel_callback(self, cancel_request) -> CancelResponse:
        self.get_logger().info("Got cancel request")
        return CancelResponse.ACCEPT


class TestActionClient(Node):
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
        super().__init__("test_message_publisher")
        self.publisher = self.create_publisher(String, topic, 10)
        self.timer = self.create_timer(0.1, self.publish_message)

    def publish_message(self) -> None:
        msg = String()
        msg.data = "Hello, ROS2!"
        self.publisher.publish(msg)


class TransformPublisher(Node):
    def __init__(self, topic: str):
        super().__init__("test_transform_publisher")
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
    # First shutdown executors
    for executor in executors:
        executor.shutdown()
    # Small delay to allow executors to finish pending operations
    time.sleep(0.5)
    # Then join threads with a timeout
    for thread in threads:
        thread.join(timeout=2.0)


@pytest.fixture(scope="function")
def ros_setup() -> Generator[None, None, None]:
    rclpy.init()
    yield
    rclpy.shutdown()
