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
from multiprocessing import Pool
from typing import Any, List
from unittest.mock import MagicMock

import pytest
from nav2_msgs.action import NavigateToPose
from rai.communication.ros2 import ROS2Connector
from rai.communication.ros2.runtime import ROS2Runtime
from rclpy.callback_groups import (
    CallbackGroup,
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from std_msgs.msg import String
from std_srvs.srv import SetBool

from .helpers import (
    MessagePublisher,
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


def test_ros2_connector_publish(ros_setup: None, request: pytest.FixtureRequest):
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    message_receiver = MessageSubscriber(topic_name)
    executors, threads = multi_threaded_spinner([message_receiver])
    connector = ROS2Connector()
    try:
        connector.sync.publish(topic_name, String(data="Hello, world!"))
        time.sleep(1)  # wait for the message to be received
        assert message_receiver.received_messages == [String(data="Hello, world!")]
    finally:
        connector.close()
        shutdown_executors_and_threads(executors, threads)


def test_ros2_connector_receive(ros_setup: None, request: pytest.FixtureRequest):
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    message_publisher = MessagePublisher(topic_name)
    executors, threads = multi_threaded_spinner([message_publisher])
    connector = ROS2Connector()
    try:
        message = connector.sync.receive(topic_name, timeout=1.0)
        assert message == String(data="Hello, ROS2!")
    finally:
        connector.close()
        shutdown_executors_and_threads(executors, threads)


def call_helper(service_name: str, connector: ROS2Connector, timeout: float = 5.0):
    response = connector.sync.call(
        service_name, SetBool.Request(data=True), timeout=timeout
    )
    assert response == SetBool.Response(success=True, message="Test service called")


@pytest.mark.parametrize(
    "callback_group",
    [MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()],
    ids=["MutuallyExclusiveCallbackGroup", "ReentrantCallbackGroup"],
)
def test_ros2_connector_call(
    ros_setup: None, request: pytest.FixtureRequest, callback_group: CallbackGroup
):
    service_name = f"{request.node.originalname}_service"  # type: ignore
    message_receiver = ServiceServer(service_name, callback_group)
    executors, threads = multi_threaded_spinner([message_receiver])
    connector = ROS2Connector()
    try:
        call_helper(service_name, connector)
    finally:
        connector.close()
        shutdown_executors_and_threads(executors, threads)


@pytest.mark.parametrize(
    "callback_group",
    [MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()],
    ids=["MutuallyExclusiveCallbackGroup", "ReentrantCallbackGroup"],
)
def test_ros2_connector_call_multiple_calls(
    ros_setup: None, request: pytest.FixtureRequest, callback_group: CallbackGroup
):
    service_name = f"{request.node.originalname}_service"  # type: ignore
    message_receiver = ServiceServer(service_name, callback_group)
    executors, threads = multi_threaded_spinner([message_receiver])
    connector = ROS2Connector()
    try:
        for _ in range(3):
            call_helper(service_name, connector)
    finally:
        connector.close()
        shutdown_executors_and_threads(executors, threads)


@pytest.mark.parametrize(
    "callback_group",
    [MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()],
    ids=["MutuallyExclusiveCallbackGroup", "ReentrantCallbackGroup"],
)
def test_ros2_connector_call_multiple_calls_at_the_same_time_threading(
    ros_setup: None, request: pytest.FixtureRequest, callback_group: CallbackGroup
):
    service_name = f"{request.node.originalname}_service"  # type: ignore
    message_receiver = ServiceServer(service_name, callback_group)
    executors, threads = multi_threaded_spinner([message_receiver])
    connector = ROS2Connector()
    errors: List[BaseException] = []

    def call() -> None:
        # The handler sleeps up to 1 s; with a mutually exclusive group the 10
        # calls run one after another, so the last answers after up to 10 s.
        try:
            call_helper(service_name, connector, timeout=15.0)
        except BaseException as e:
            errors.append(e)

    try:
        service_threads: List[threading.Thread] = []
        for _ in range(10):
            thread = threading.Thread(target=call)
            service_threads.append(thread)
            thread.start()

        for thread in service_threads:
            thread.join()
        assert not errors, errors
    finally:
        connector.close()
        shutdown_executors_and_threads(executors, threads)


@pytest.mark.skip(reason="As of now, multiprocessing does not work with ROS2Connector.")
@pytest.mark.parametrize(
    "callback_group",
    [MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()],
    ids=["MutuallyExclusiveCallbackGroup", "ReentrantCallbackGroup"],
)
def test_ros2_connector_call_multiple_calls_at_the_same_time_multiprocessing(
    ros_setup: None, request: pytest.FixtureRequest, callback_group: CallbackGroup
):
    service_name = f"{request.node.originalname}_service"  # type: ignore
    message_receiver = ServiceServer(service_name, callback_group)
    executors, threads = multi_threaded_spinner([message_receiver])
    connector = ROS2Connector()
    try:
        with Pool(10) as pool:
            pool.map(lambda _: call_helper(service_name, connector), range(10))
    finally:
        connector.close()
        shutdown_executors_and_threads(executors, threads)


def test_ros2_connector_send_goal(ros_setup: None, request: pytest.FixtureRequest):
    action_name = f"{request.node.originalname}_action"  # type: ignore
    action_server = TestActionServer(action_name)
    executors, threads = multi_threaded_spinner([action_server])
    connector = ROS2Connector()
    try:
        handle = connector.sync.send_goal(action_name, NavigateToPose.Goal())
        assert handle.id
    finally:
        connector.close()
        shutdown_executors_and_threads(executors, threads)


def test_ros2_connector_send_goal_and_cancel(
    ros_setup: None, request: pytest.FixtureRequest
):
    action_name = f"{request.node.originalname}_action"  # type: ignore
    action_server = TestActionServer(action_name)
    executors, threads = multi_threaded_spinner([action_server])
    connector = ROS2Connector()
    feedbacks: List[Any] = []
    try:
        handle = connector.sync.send_goal(
            action_name, NavigateToPose.Goal(), on_feedback=feedbacks.append
        )
        assert handle.id
        feedbacks_before = feedbacks
        handle.sync.cancel()
        time.sleep(0.2)
        feedbacks_after = feedbacks
        assert len(feedbacks_after) == len(feedbacks_before)
    finally:
        connector.close()
        shutdown_executors_and_threads(executors, threads)


def test_ros2_connector_send_goal_erroneous_callback(
    ros_setup: None, request: pytest.FixtureRequest
):
    action_name = f"{request.node.originalname}_action"  # type: ignore
    action_server = TestActionServer(action_name)
    executors, threads = multi_threaded_spinner([action_server])
    connector = ROS2Connector()
    try:
        handle = connector.sync.send_goal(
            action_name,
            NavigateToPose.Goal(),
            on_feedback=lambda feedback: 1 / 0,  # trigger ZeroDivisionError
        )
        assert handle.id
    finally:
        connector.close()
        shutdown_executors_and_threads(executors, threads)


def test_ros2_connector_create_service(ros_setup: None, request: pytest.FixtureRequest):
    connector = ROS2Connector()
    mock_callback = MagicMock()
    mock_callback.return_value = SetBool.Response()

    try:
        handle = connector.create_service(
            service_name="set_bool",
            service_type="std_srvs/srv/SetBool",
            on_request=mock_callback,
        )
        assert handle is not None
        # Give the connector's executor time to register the service in the ROS2 graph
        time.sleep(0.005)
        service_client = TestServiceClient()
        executors, threads = multi_threaded_spinner([service_client])
        service_client.send_request()
        time.sleep(0.2)
        assert mock_callback.called
    finally:
        connector.close()
        shutdown_executors_and_threads(executors, threads)


def test_ros2_connector_action_call(ros_setup: None, request: pytest.FixtureRequest):
    action_name = "navigate_to_pose"
    connector = ROS2Connector()
    mock_callback = MagicMock()
    mock_callback.return_value = NavigateToPose.Result()

    try:
        action_server_handle = connector.create_action(
            action_name,
            generate_feedback_callback=mock_callback,
            action_type="nav2_msgs/action/NavigateToPose",
        )
        assert action_server_handle is not None
        action_client = TestActionClient()
        executors, threads = multi_threaded_spinner([action_client])
        action_client.send_goal()
        time.sleep(0.02)
        assert mock_callback.called
    finally:
        connector.close()
        shutdown_executors_and_threads(executors, threads)


def test_ros2_connector_unique_names(ros_setup: None):
    """Default connectors share one node; private runtimes get unique names."""
    connector1 = ROS2Connector(runtime=ROS2Runtime())
    connector2 = ROS2Connector(runtime=ROS2Runtime())
    try:
        name1 = connector1.node_name
        name2 = connector2.node_name
        assert name1 != name2
        assert name1.startswith("/rai_") and name2.startswith("/rai_")
    finally:
        connector1.close()
        connector2.close()


def test_ros2_connector_parameter(ros_setup: None):
    """parameter() declares a default on first use; --ros-args overrides win."""
    runtime = ROS2Runtime(cli_args=["--ros-args", "-p", "rate:=2.5"])
    with ROS2Connector(runtime=runtime) as connector:
        assert connector.declare_parameter("rate", 1.0) == 2.5
        assert connector.declare_parameter("frame", "map") == "map"
        assert connector.declare_parameter("frame", "odom") == "map"  # declared once
