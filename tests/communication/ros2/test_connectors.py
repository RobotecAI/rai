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
from PIL import Image
from pydub import AudioSegment
from rai.communication.ros2 import (
    ROS2Connector,
    ROS2HRIConnector,
    ROS2HRIMessage,
    ROS2Message,
)
from rclpy.callback_groups import (
    CallbackGroup,
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from std_msgs.msg import String
from std_srvs.srv import SetBool

from .helpers import (
    HRIMessageSubscriber,
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


def test_ros2_connector_send_message(ros_setup: None, request: pytest.FixtureRequest):
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    message_receiver = MessageSubscriber(topic_name)
    executors, threads = multi_threaded_spinner([message_receiver])
    connector = ROS2Connector()
    try:
        message = ROS2Message(
            payload={"data": "Hello, world!"},
            metadata={"msg_type": "std_msgs/msg/String"},
        )
        connector.send_message(
            message=message, target=topic_name, msg_type="std_msgs/msg/String"
        )
        time.sleep(1)  # wait for the message to be received
        assert message_receiver.received_messages == [String(data="Hello, world!")]
    finally:
        connector.shutdown()
        shutdown_executors_and_threads(executors, threads)


def test_ros2_connector_receive_message(
    ros_setup: None, request: pytest.FixtureRequest
):
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    message_publisher = MessagePublisher(topic_name)
    executors, threads = multi_threaded_spinner([message_publisher])
    connector = ROS2Connector()
    try:
        message = connector.receive_message(topic_name, timeout_sec=1.0)
        assert message.payload == String(data="Hello, ROS2!")
    finally:
        connector.shutdown()
        shutdown_executors_and_threads(executors, threads)


def service_call_helper(service_name: str, connector: ROS2Connector):
    message = ROS2Message(payload={"data": True})
    response = connector.service_call(
        message, target=service_name, msg_type="std_srvs/srv/SetBool"
    )
    assert response.payload == SetBool.Response(
        success=True, message="Test service called"
    )


@pytest.mark.parametrize(
    "callback_group",
    [MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()],
    ids=["MutuallyExclusiveCallbackGroup", "ReentrantCallbackGroup"],
)
def test_ros2_connector_service_call(
    ros_setup: None, request: pytest.FixtureRequest, callback_group: CallbackGroup
):
    service_name = f"{request.node.originalname}_service"  # type: ignore
    message_receiver = ServiceServer(service_name, callback_group)
    executors, threads = multi_threaded_spinner([message_receiver])
    connector = ROS2Connector()
    try:
        service_call_helper(service_name, connector)
    finally:
        connector.shutdown()
        shutdown_executors_and_threads(executors, threads)


@pytest.mark.parametrize(
    "callback_group",
    [MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()],
    ids=["MutuallyExclusiveCallbackGroup", "ReentrantCallbackGroup"],
)
def test_ros2_connector_service_call_multiple_calls(
    ros_setup: None, request: pytest.FixtureRequest, callback_group: CallbackGroup
):
    service_name = f"{request.node.originalname}_service"  # type: ignore
    message_receiver = ServiceServer(service_name, callback_group)
    executors, threads = multi_threaded_spinner([message_receiver])
    connector = ROS2Connector()
    try:
        for _ in range(3):
            service_call_helper(service_name, connector)
    finally:
        connector.shutdown()
        shutdown_executors_and_threads(executors, threads)


@pytest.mark.parametrize(
    "callback_group",
    [MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()],
    ids=["MutuallyExclusiveCallbackGroup", "ReentrantCallbackGroup"],
)
def test_ros2_connector_service_call_multiple_calls_at_the_same_time_threading(
    ros_setup: None, request: pytest.FixtureRequest, callback_group: CallbackGroup
):
    service_name = f"{request.node.originalname}_service"  # type: ignore
    message_receiver = ServiceServer(service_name, callback_group)
    executors, threads = multi_threaded_spinner([message_receiver])
    connector = ROS2Connector()
    try:
        service_threads: List[threading.Thread] = []
        for _ in range(10):
            thread = threading.Thread(
                target=service_call_helper, args=(service_name, connector)
            )
            service_threads.append(thread)
            thread.start()

        for thread in service_threads:
            thread.join()
    finally:
        connector.shutdown()
        shutdown_executors_and_threads(executors, threads)


@pytest.mark.skip(reason="As of now, multiprocessing does not work with ROS2Connector.")
@pytest.mark.parametrize(
    "callback_group",
    [MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()],
    ids=["MutuallyExclusiveCallbackGroup", "ReentrantCallbackGroup"],
)
def test_ros2_connector_service_call_multiple_calls_at_the_same_time_multiprocessing(
    ros_setup: None, request: pytest.FixtureRequest, callback_group: CallbackGroup
):
    service_name = f"{request.node.originalname}_service"  # type: ignore
    message_receiver = ServiceServer(service_name, callback_group)
    executors, threads = multi_threaded_spinner([message_receiver])
    connector = ROS2Connector()
    try:
        with Pool(10) as pool:
            pool.map(lambda _: service_call_helper(service_name, connector), range(10))
    finally:
        connector.shutdown()
        shutdown_executors_and_threads(executors, threads)


def test_ros2_connector_send_goal(ros_setup: None, request: pytest.FixtureRequest):
    action_name = f"{request.node.originalname}_action"  # type: ignore
    action_server = TestActionServer(action_name)
    executors, threads = multi_threaded_spinner([action_server])
    connector = ROS2Connector()
    try:
        message = ROS2Message(
            payload={},
        )
        handle = connector.start_action(
            action_data=message,
            target=action_name,
            msg_type="nav2_msgs/action/NavigateToPose",
        )
        assert handle is not None
    finally:
        connector.shutdown()
        shutdown_executors_and_threads(executors, threads)


def test_ros2_connector_send_goal_and_terminate_action(
    ros_setup: None, request: pytest.FixtureRequest
):
    action_name = f"{request.node.originalname}_action"  # type: ignore
    action_server = TestActionServer(action_name)
    executors, threads = multi_threaded_spinner([action_server])
    connector = ROS2Connector()
    feedbacks: List[Any] = []
    try:
        message = ROS2Message(payload={})
        handle = connector.start_action(
            action_data=message,
            target=action_name,
            on_feedback=lambda feedback: feedbacks.append(feedback),
            msg_type="nav2_msgs/action/NavigateToPose",
        )
        assert handle is not None
        feedbacks_before = feedbacks
        connector.terminate_action(handle)
        time.sleep(0.2)
        feedbacks_after = feedbacks
        assert len(feedbacks_after) == len(feedbacks_before)
    finally:
        connector.shutdown()
        shutdown_executors_and_threads(executors, threads)


def test_ros2_connector_send_goal_erroneous_callback(
    ros_setup: None, request: pytest.FixtureRequest
):
    action_name = f"{request.node.originalname}_action"  # type: ignore
    action_server = TestActionServer(action_name)
    executors, threads = multi_threaded_spinner([action_server])
    connector = ROS2Connector()
    try:
        message = ROS2Message(payload={})
        handle = connector.start_action(
            action_data=message,
            target=action_name,
            on_feedback=lambda feedback: 1 / 0,  # trigger ZeroDivisionError
            msg_type="nav2_msgs/action/NavigateToPose",
        )
        assert handle is not None
    finally:
        connector.shutdown()
        shutdown_executors_and_threads(executors, threads)


def test_ros2hri_default_message_publish(
    ros_setup: None, request: pytest.FixtureRequest
):
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    connector = ROS2HRIConnector()
    hri_message_receiver = HRIMessageSubscriber(topic_name)
    executors, threads = multi_threaded_spinner([hri_message_receiver])

    try:
        images = [Image.new("RGB", (100, 100), color="red")]
        audios = [AudioSegment.silent(duration=1000)]
        text = "Hello, HRI!"
        message = ROS2HRIMessage(
            text=text,
            images=images,
            audios=audios,
            message_author="ai",
            communication_id=ROS2HRIMessage.generate_conversation_id(),
        )
        connector.send_message(message, target=topic_name)
        time.sleep(1)  # wait for the message to be received

        assert len(hri_message_receiver.received_messages) > 0
        recreated_message = ROS2HRIMessage.from_ros2(
            hri_message_receiver.received_messages[0], message_author="ai"
        )

        assert message.text == recreated_message.text
        assert message.message_author == recreated_message.message_author
        assert len(message.images) == len(recreated_message.images)
        assert len(message.audios) == len(recreated_message.audios)
        assert all(
            [
                image_a == image_b
                for image_a, image_b in zip(message.images, recreated_message.images)
            ]
        )
        assert all(
            [
                audio_a == audio_b
                for audio_a, audio_b in zip(message.audios, recreated_message.audios)
            ]
        )
    finally:
        connector.shutdown()
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
        connector.shutdown()
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
        shutdown_executors_and_threads(executors, threads)


def test_ros2_connector_unique_names(ros_setup: None):
    connector1 = ROS2Connector()
    connector2 = ROS2Connector()
    try:
        name1 = connector1.node.get_name()
        name2 = connector2.node.get_name()
        assert name1 != name2
        assert "rai_ros2_connector_" in name1
        assert "rai_ros2_connector_" in name2
    finally:
        connector1.shutdown()
        connector2.shutdown()
