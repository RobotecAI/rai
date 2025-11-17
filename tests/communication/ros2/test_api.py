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

import logging
import threading
import time
from multiprocessing import Pool
from typing import Any, List
from unittest.mock import MagicMock

import pytest
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from nav2_msgs.action import NavigateToPose
from rai.communication.ros2.api import (
    ROS2ActionAPI,
    ROS2ServiceAPI,
    ROS2TopicAPI,
)
from rclpy.callback_groups import (
    CallbackGroup,
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import SetBool

from .helpers import (
    MessageSubscriber,
    ServiceServer,
    TestActionClient,
    TestActionServer,
    TestServiceClient,
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
    message_receiver = MessageSubscriber(topic_name)
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
    message_receiver = MessageSubscriber(topic_name)
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
    message_receiver = MessageSubscriber(topic_name)
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
    message_receiver = MessageSubscriber(topic_name)
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


def invoke_set_bool_service(
    service_name: str, service_api: ROS2ServiceAPI, reuse_client: bool = True
):
    response = service_api.call_service(
        service_name,
        service_type="std_srvs/srv/SetBool",
        request={"data": True},
        reuse_client=reuse_client,
    )
    assert response.success
    assert response.message == "Test service called"


@pytest.mark.parametrize(
    "callback_group",
    [MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()],
    ids=["MutuallyExclusiveCallbackGroup", "ReentrantCallbackGroup"],
)
def test_ros2_service_single_call(
    ros_setup: None, request: pytest.FixtureRequest, callback_group: CallbackGroup
) -> None:
    service_name = f"{request.node.originalname}_service"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    service_server = ServiceServer(service_name, callback_group)
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([service_server, node])

    try:
        service_api = ROS2ServiceAPI(node)
        invoke_set_bool_service(service_name, service_api)
    finally:
        shutdown_executors_and_threads(executors, threads)


@pytest.mark.parametrize(
    "callback_group",
    [MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()],
    ids=["MutuallyExclusiveCallbackGroup", "ReentrantCallbackGroup"],
)
def test_ros2_service_multiple_calls(
    ros_setup: None, request: pytest.FixtureRequest, callback_group: CallbackGroup
) -> None:
    service_name = f"{request.node.originalname}_service"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    service_server = ServiceServer(service_name, callback_group)
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([service_server, node])

    try:
        service_api = ROS2ServiceAPI(node)
        for _ in range(3):
            invoke_set_bool_service(service_name, service_api, reuse_client=False)
    finally:
        shutdown_executors_and_threads(executors, threads)


@pytest.mark.parametrize(
    "callback_group",
    [MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()],
    ids=["MutuallyExclusiveCallbackGroup", "ReentrantCallbackGroup"],
)
def test_ros2_service_multiple_calls_with_reused_client(
    ros_setup: None, request: pytest.FixtureRequest, callback_group: CallbackGroup
) -> None:
    service_name = f"{request.node.originalname}_service"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    service_server = ServiceServer(service_name, callback_group)
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([service_server, node])

    try:
        service_api = ROS2ServiceAPI(node)
        for _ in range(3):
            invoke_set_bool_service(service_name, service_api, reuse_client=True)
        assert service_api.release_client(service_name), "Client not released"
    finally:
        shutdown_executors_and_threads(executors, threads)


@pytest.mark.parametrize(
    "callback_group",
    [MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()],
    ids=["MutuallyExclusiveCallbackGroup", "ReentrantCallbackGroup"],
)
def test_ros2_service_multiple_calls_at_the_same_time_threading(
    ros_setup: None, request: pytest.FixtureRequest, callback_group: CallbackGroup
) -> None:
    service_name = f"{request.node.originalname}_service"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    service_server = ServiceServer(service_name, callback_group)
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([service_server, node])

    try:
        service_api = ROS2ServiceAPI(node)
        service_threads: List[threading.Thread] = []
        for _ in range(10):
            thread = threading.Thread(
                target=invoke_set_bool_service, args=(service_name, service_api)
            )
            service_threads.append(thread)
            thread.start()

        # Wait for all service threads to complete
        for thread in service_threads:
            thread.join()
    finally:
        shutdown_executors_and_threads(executors, threads)


@pytest.mark.skip(reason="As of now, multiprocessing does not work with ROS2API")
@pytest.mark.parametrize(
    "callback_group",
    [MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()],
    ids=["MutuallyExclusiveCallbackGroup", "ReentrantCallbackGroup"],
)
def test_ros2_service_multiple_calls_at_the_same_time_multiprocessing(
    ros_setup: None, request: pytest.FixtureRequest, callback_group: CallbackGroup
) -> None:
    service_name = f"{request.node.originalname}_service"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    service_server = ServiceServer(service_name, callback_group)
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([service_server, node])

    try:
        service_api = ROS2ServiceAPI(node)
        with Pool(10) as pool:
            pool.map(
                lambda _: invoke_set_bool_service(service_name, service_api), range(10)
            )
    finally:
        shutdown_executors_and_threads(executors, threads)


@pytest.mark.parametrize(
    "callback_group",
    [MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()],
    ids=["MutuallyExclusiveCallbackGroup", "ReentrantCallbackGroup"],
)
def test_ros2_service_single_call_wrong_service_type(
    ros_setup: None, request: pytest.FixtureRequest, callback_group: CallbackGroup
) -> None:
    service_name = f"{request.node.originalname}_service"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    service_server = ServiceServer(service_name, callback_group)
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


@pytest.mark.parametrize(
    "callback_group",
    [MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()],
    ids=["MutuallyExclusiveCallbackGroup", "ReentrantCallbackGroup"],
)
def test_ros2_service_single_call_wrong_service_content(
    ros_setup: None, request: pytest.FixtureRequest, callback_group: CallbackGroup
) -> None:
    service_name = f"{request.node.originalname}_service"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    service_server = ServiceServer(service_name, callback_group)
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


@pytest.mark.parametrize(
    "callback_group",
    [MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()],
    ids=["MutuallyExclusiveCallbackGroup", "ReentrantCallbackGroup"],
)
def test_ros2_service_single_call_wrong_service_name(
    ros_setup: None, request: pytest.FixtureRequest, callback_group: CallbackGroup
) -> None:
    service_name = f"{request.node.originalname}_service"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    service_server = ServiceServer(service_name, callback_group)
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([service_server, node])

    try:
        service_api = ROS2ServiceAPI(node)
        with pytest.raises(ValueError):
            service_api.call_service(
                f"{service_name}/wrong_service_name",
                service_type="std_srvs/srv/SetBool",
                request={"data": True},
                timeout_sec=1.0,
            )
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_action_send_goal(ros_setup: None, request: pytest.FixtureRequest) -> None:
    action_name = f"{request.node.originalname}_action"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    action_server = TestActionServer(action_name)
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
    action_server = TestActionServer(action_name)
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
    action_server = TestActionServer(action_name)
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
    action_server = TestActionServer(action_name)
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
    action_server = TestActionServer(action_name)
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


def test_ros2_action_feedback_callback_exception_handling(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    action_name = f"{request.node.originalname}_action"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    action_server = TestActionServer(action_name)
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([action_server])
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin)
    thread.start()
    action_api = ROS2ActionAPI(node)

    error_callback_called = False

    def failing_feedback_callback(feedback_msg: Any) -> None:
        nonlocal error_callback_called
        error_callback_called = True
        raise ValueError("Test exception in feedback callback")

    mock_logger = MagicMock()
    action_api._logger = mock_logger

    try:
        accepted, handle = action_api.send_goal(
            action_name,
            "nav2_msgs/action/NavigateToPose",
            {},
            feedback_callback=failing_feedback_callback,
        )
        assert accepted
        assert handle != ""

        wait_time = 0.5
        start_time = time.perf_counter()
        while not action_api.is_goal_done(handle):
            time.sleep(0.01)
            if time.perf_counter() - start_time > wait_time:
                raise TimeoutError("Goal not done")

        result = action_api.get_result(handle)
        assert result.status == GoalStatus.STATUS_SUCCEEDED

        time.sleep(0.1)
        assert error_callback_called, "Error callback should have been called"
        mock_logger.error.assert_called()
        error_call_args = mock_logger.error.call_args[0][0]
        assert "Error in feedback callback" in error_call_args
        assert "Test exception in feedback callback" in error_call_args
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_action_send_goal_terminate_goal(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    action_name = f"{request.node.originalname}_action"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    action_server = TestActionServer(action_name)
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


def test_ros2_action_send_goal_timeout(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    """Test that send_goal returns (False, "") when goal send times out."""
    action_name = f"{request.node.originalname}_nonexistent_action"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([node])

    try:
        action_api = ROS2ActionAPI(node)
        # Use very short timeout - server doesn't exist so wait_for_server will timeout
        accepted, handle = action_api.send_goal(
            action_name, "nav2_msgs/action/NavigateToPose", {}, timeout_sec=0.01
        )

        assert not accepted
        assert handle == ""
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_action_send_goal_timeout_logs_warning(
    ros_setup: None, request: pytest.FixtureRequest, caplog
) -> None:
    """Test that send_goal timeout logs a warning via get_future_result."""
    action_name = f"{request.node.originalname}_action"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    action_server = TestActionServer(action_name)
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([action_server, node])

    try:
        action_api = ROS2ActionAPI(node)
        # Wait for server first
        time.sleep(0.1)

        # Ensure logger propagates to root logger for caplog to capture
        logger = logging.getLogger("rai.communication.ros2.ros_async")
        logger.propagate = True

        # Now use a very short timeout for goal send to trigger timeout
        with caplog.at_level(logging.WARNING, logger=logger.name):
            accepted, handle = action_api.send_goal(
                action_name, "nav2_msgs/action/NavigateToPose", {}, timeout_sec=0.001
            )

        # Should timeout and log warning
        assert not accepted
        assert handle == ""
        assert any("Future timed out" in record.message for record in caplog.records)
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_action_send_goal_exception_propagates(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    """Test that exceptions from send_goal_future propagate correctly via get_future_result."""
    action_name = f"{request.node.originalname}_action"  # type: ignore
    node_name = f"{request.node.originalname}_node"  # type: ignore
    action_server = TestActionServer(action_name)
    node = Node(node_name)
    executors, threads = multi_threaded_spinner([action_server, node])

    try:
        action_api = ROS2ActionAPI(node)
        # Invalid action type will raise during import, not in future
        # This tests that exceptions are properly propagated through get_future_result
        with pytest.raises((ValueError, ImportError, AttributeError)):
            action_api.send_goal(action_name, "invalid/action/Type", {})
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_ros2_create_action_server(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    action_name = "navigate_to_pose"
    node_name = f"{request.node.originalname}_node"  # type: ignore
    node = Node(node_name)
    mock_callback = MagicMock()
    mock_callback.return_value = NavigateToPose.Result()

    try:
        action_api = ROS2ActionAPI(node)
        action_server_handle = action_api.create_action_server(
            "nav2_msgs/action/NavigateToPose",
            action_name,
            execute_callback=mock_callback,
        )
        assert action_server_handle is not None
        action_client = TestActionClient()
        executors, threads = multi_threaded_spinner([node, action_client])
        action_client.send_goal()
        time.sleep(0.01)
    finally:
        shutdown_executors_and_threads(executors, threads)
        assert mock_callback.called


def test_ros2_create_create_service(ros_setup: None, request: pytest.FixtureRequest):
    service_name = "set_bool"
    node_name = f"{request.node.originalname}_node"
    node = Node(node_name)
    mock_callback = MagicMock()
    mock_callback.return_value = SetBool.Response()

    try:
        service_api = ROS2ServiceAPI(node)
        service_server_handle = service_api.create_service(
            service_name,
            "std_srvs/srv/SetBool",
            callback=mock_callback,
        )
        assert service_server_handle is not None
        client = TestServiceClient()
        executors, threads = multi_threaded_spinner([node, client])
        client.send_request()
        time.sleep(0.01)
    finally:
        shutdown_executors_and_threads(executors, threads)
        assert mock_callback.called
