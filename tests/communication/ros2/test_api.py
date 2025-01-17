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
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionServer
from rclpy.action.client import ClientGoalHandle
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool

from rai.communication.ros2.api import ROS2ActionAPI, ROS2ServiceAPI, ROS2TopicAPI


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
        )

    def handle_test_action(
        self, goal_handle: ClientGoalHandle
    ) -> NavigateToPose.Result:

        for i in range(1, 11):
            feedback_msg = NavigateToPose.Feedback(distance_remaining=10.0 / i)
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(0.01)

        goal_handle.succeed()

        result = NavigateToPose.Result()
        result.error_code = NavigateToPose.Result.NONE
        return result


class MessagePublisher(Node):
    def __init__(self, topic: str):
        super().__init__("test_message_publisher")
        self.publisher = self.create_publisher(String, topic, 10)
        self.timer = self.create_timer(0.1, self.publish_message)

    def publish_message(self) -> None:
        msg = String()
        msg.data = "Hello, ROS2!"
        self.publisher.publish(msg)


def single_threaded_spinner(
    nodes: List[Node],
) -> Tuple[List[SingleThreadedExecutor], List[threading.Thread]]:
    executors: List[SingleThreadedExecutor] = []
    executor_threads: List[threading.Thread] = []
    for node in nodes:
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executors.append(executor)
    for executor in executors:
        executor_thread = threading.Thread(target=executor.spin)
        executor_thread.daemon = True
        executor_thread.start()
        executor_threads.append(executor_thread)
    return executors, executor_threads


def shutdown_executors_and_threads(
    executors: List[SingleThreadedExecutor], threads: List[threading.Thread]
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


def test_ros2_single_message_publish(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    message_receiver = MessageReceiver(topic_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([message_receiver, node])

    try:
        topic_api = ROS2TopicAPI(node)
        topic_api.publish(
            topic_name,
            {"data": "Hello, ROS2!"},
            msg_type="std_msgs/msg/String",
        )
        time.sleep(1)
        assert len(message_receiver.received_messages) == 1
        assert message_receiver.received_messages[0].data == "Hello, ROS2!"
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_single_message_publish_wrong_msg_type(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    message_receiver = MessageReceiver(topic_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([message_receiver, node])

    try:
        topic_api = ROS2TopicAPI(node)
        with pytest.raises(AttributeError):
            topic_api.publish(
                topic_name,
                {"data": "Hello, ROS2!"},
                msg_type="std_msgs/msg/NotExistingMessage",
            )
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_single_message_publish_wrong_msg_content(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    message_receiver = MessageReceiver(topic_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([message_receiver, node])

    try:
        topic_api = ROS2TopicAPI(node)
        with pytest.raises(AttributeError):
            topic_api.publish(
                topic_name,
                {"NotExistingField": "Hello, ROS2!"},
                msg_type="std_msgs/msg/String",
            )
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_single_message_publish_wrong_qos_setup(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    message_receiver = MessageReceiver(topic_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([message_receiver, node])

    try:
        topic_api = ROS2TopicAPI(node)
        with pytest.raises(ValueError):
            topic_api.publish(
                topic_name,
                {"data": "Hello, ROS2!"},
                msg_type="std_msgs/msg/String",
                auto_qos_matching=False,
                qos_profile=None,
            )
    finally:
        shutdown_executors_and_threads(executors, threads)


@pytest.mark.xfail(
    reason="Test expected to fail: ROS2 node discovery is asynchronous and the current implementation "
    "doesn't wait for topic discovery. "
    "TODO: Implement a proper discovery mechanism with timeout "
    "to ensure reliable topic communication."
)
def test_ros2_single_message_receive_no_discovery_time(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    message_publisher = MessagePublisher(topic_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([message_publisher, node])

    try:
        topic_api = ROS2TopicAPI(node)
        msg = topic_api.receive(
            topic_name,
            msg_type="std_msgs/msg/String",
            timeout_sec=3.0,
            auto_topic_type=False,
        )
        assert msg.data == "Hello, ROS2!"
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_single_message_receive_wrong_msg_type(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    message_publisher = MessagePublisher(topic_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([message_publisher, node])

    try:
        topic_api = ROS2TopicAPI(node)
        with pytest.raises(AttributeError):
            topic_api.receive(
                topic_name,
                msg_type="std_msgs/msg/NotExistingMessage",
                auto_topic_type=False,
            )
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_single_message_receive_wrong_topic_name(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    message_publisher = MessagePublisher(topic_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([message_publisher, node])

    try:
        topic_api = ROS2TopicAPI(node)
        with pytest.raises(ValueError):
            topic_api.receive(
                f"{topic_name}/wrong_topic_name",
                msg_type="std_msgs/msg/String",
                auto_topic_type=False,
            )
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_service_single_call(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    service_name = f"{request.node.originalname}_service"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    service_server = ServiceServer(service_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([service_server, node])

    try:
        service_api = ROS2ServiceAPI(node)
        response = service_api.call_service(
            service_name,
            service_type="std_srvs/srv/SetBool",
            request={"data": True},
        )
        assert response.success
        assert response.message == "Test service called"
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_service_single_call_wrong_service_type(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    service_name = f"{request.node.originalname}_service"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    service_server = ServiceServer(service_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([service_server, node])

    try:
        service_api = ROS2ServiceAPI(node)
        with pytest.raises(AttributeError):
            service_api.call_service(
                service_name,
                service_type="std_srvs/srv/NotExistingService",
                request={"data": True},
            )
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_service_single_call_wrong_service_content(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    service_name = f"{request.node.originalname}_service"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    service_server = ServiceServer(service_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([service_server, node])

    try:
        service_api = ROS2ServiceAPI(node)
        with pytest.raises(AttributeError):
            service_api.call_service(
                service_name,
                service_type="std_srvs/srv/SetBool",
                request={"NotExistingField": True},
            )
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_service_single_call_wrong_service_name(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    service_name = f"{request.node.originalname}_service"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    service_server = ServiceServer(service_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([service_server, node])

    try:
        service_api = ROS2ServiceAPI(node)
        with pytest.raises(ValueError):
            service_api.call_service(
                f"{service_name}/wrong_service_name",
                service_type="std_srvs/srv/SetBool",
                request={"data": True},
            )
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_action_send_goal(ros_setup: None, request: pytest.FixtureRequest) -> None:
    action_name = f"{request.node.originalname}_action"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    action_server = ActionServer_(action_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([action_server, node])

    try:
        action_api = ROS2ActionAPI(node)
        accepted, handle = action_api.send_goal(
            action_name, "nav2_msgs/action/NavigateToPose", {}
        )

        assert accepted
        assert handle != ""
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_action_send_goal_get_result(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    action_name = f"{request.node.originalname}_action"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    action_server = ActionServer_(action_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([action_server, node])

    try:
        action_api = ROS2ActionAPI(node)
        accepted, handle = action_api.send_goal(
            action_name, "nav2_msgs/action/NavigateToPose", {}
        )
        import time

        wait_time = 1.0
        start_time = time.perf_counter()
        while not action_api.is_goal_done(handle):
            time.sleep(0.01)
            if time.perf_counter() - start_time > wait_time:
                raise TimeoutError("Goal not done")
        result = action_api.get_result(handle)

        assert result.status == GoalStatus.STATUS_SUCCEEDED
        assert accepted
        assert handle != ""
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_action_send_goal_wrong_action_type(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    action_name = f"{request.node.originalname}_action"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    action_server = ActionServer_(action_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([action_server, node])

    try:
        action_api = ROS2ActionAPI(node)
        with pytest.raises(ModuleNotFoundError):
            action_api.send_goal(
                action_name, "nav2_msgs/action/NavigateToPose/WrongActionType", {}
            )
        with pytest.raises(AttributeError):
            action_api.send_goal(action_name, "nav2_msgs/action/NavigateToPoses", {})
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_action_send_goal_wrong_action_name(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    action_name = f"{request.node.originalname}_action"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    action_server = ActionServer_(action_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([action_server, node])

    try:
        action_api = ROS2ActionAPI(node)
        accepted, handle = action_api.send_goal(
            "WrongActionName", "nav2_msgs/action/NavigateToPose", {}
        )
        assert not accepted
        assert handle == ""
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_action_send_goal_get_feedback(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    action_name = f"{request.node.originalname}_action"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    action_server = ActionServer_(action_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([action_server])
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin)
    thread.start()
    action_api = ROS2ActionAPI(node)
    accepted, handle = action_api.send_goal(
        action_name, "nav2_msgs/action/NavigateToPose", {}
    )
    assert accepted
    assert handle != ""
    time.sleep(0.2)
    try:
        feedback = action_api.get_feedback(handle)
        distances = [x.distance_remaining for x in feedback]
        assert sorted(distances, reverse=True) == distances, "Wrong message order"
        assert len(distances) == 10, "Wrong number of messages"
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_action_send_goal_terminate_goal(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    action_name = f"{request.node.originalname}_action"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    action_server = ActionServer_(action_name)
    node = Node(node_name)
    executors, threads = single_threaded_spinner([action_server])
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin)
    thread.start()
    action_api = ROS2ActionAPI(node)
    try:
        accepted, handle = action_api.send_goal(
            action_name, "nav2_msgs/action/NavigateToPose", {}
        )
        assert accepted
        assert handle != ""
        feedbacks_before = action_api.get_feedback(handle)
        response = action_api.terminate_goal(handle)
        assert response.return_code == CancelGoal.Response.ERROR_GOAL_TERMINATED  # type: ignore
        assert action_api.is_goal_done(handle)
        feedbacks_after = action_api.get_feedback(handle)
        assert len(feedbacks_before) == len(feedbacks_after)
    finally:
        shutdown_executors_and_threads(executors, threads)
