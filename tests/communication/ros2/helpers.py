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
from typing import Generator, List, Tuple

import pytest
import rclpy
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool


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


class MessageReceiver(Node):
    def __init__(self, topic: str):
        super().__init__("test_message_receiver")
        self.subscription = self.create_subscription(
            String, topic, self.handle_test_message, 10
        )
        self.received_messages: List[String] = []

    def handle_test_message(self, msg: String) -> None:
        self.received_messages.append(msg)


class ActionServer_(Node):
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


class MessagePublisher(Node):
    def __init__(self, topic: str):
        super().__init__("test_message_publisher")
        self.publisher = self.create_publisher(String, topic, 10)
        self.timer = self.create_timer(0.1, self.publish_message)

    def publish_message(self) -> None:
        msg = String()
        msg.data = "Hello, ROS2!"
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
