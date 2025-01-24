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

import pytest
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from rai.communication.ros2.api import ROS2ActionAPI, ROS2ServiceAPI, ROS2TopicAPI
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from .helpers import ActionServer_ as ActionServer
from .helpers import (
    MessagePublisher,
    MessageReceiver,
    ServiceServer,
    multi_threaded_spinner,
    ros_setup,
    shutdown_executors_and_threads,
)

_ = ros_setup  # Explicitly use the fixture to prevent pytest warnings


def test_ros2_single_message_publish(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    message_receiver = MessageReceiver(topic_name)
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([message_receiver, node])

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
    executors, threads = multi_threaded_spinner([message_receiver, node])

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
    executors, threads = multi_threaded_spinner([message_receiver, node])

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
    executors, threads = multi_threaded_spinner([message_receiver, node])

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
    executors, threads = multi_threaded_spinner([message_publisher, node])

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
    executors, threads = multi_threaded_spinner([message_publisher, node])

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
    executors, threads = multi_threaded_spinner([message_publisher, node])

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
    executors, threads = multi_threaded_spinner([service_server, node])

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
    executors, threads = multi_threaded_spinner([service_server, node])

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
    executors, threads = multi_threaded_spinner([service_server, node])

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
    executors, threads = multi_threaded_spinner([service_server, node])

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
    action_server = ActionServer(action_name)
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([action_server, node])

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
    action_server = ActionServer(action_name)
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([action_server, node])

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
    action_server = ActionServer(action_name)
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([action_server, node])

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
    action_server = ActionServer(action_name)
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([action_server, node])

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
    action_server = ActionServer(action_name)
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([action_server])
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
    action_server = ActionServer(action_name)
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([action_server])
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
        assert response.return_code == CancelGoal.Response.ERROR_NONE  # type: ignore
        time.sleep(0.1)
        assert action_api.is_goal_done(handle)
        feedbacks_after = action_api.get_feedback(handle)
        assert len(feedbacks_before) == len(feedbacks_after)
    finally:
        shutdown_executors_and_threads(executors, threads)
