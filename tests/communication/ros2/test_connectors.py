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

import time
from typing import Any, List

import pytest
from PIL import Image
from pydub import AudioSegment
from rai.communication.ros2.connectors import (
    HRIPayload,
    ROS2ARIConnector,
    ROS2ARIMessage,
    ROS2HRIConnector,
    ROS2HRIMessage,
)
from std_msgs.msg import String
from std_srvs.srv import SetBool

from .helpers import ActionServer_ as ActionServer
from .helpers import (
    HRIMessageSubscriber,
    MessagePublisher,
    MessageReceiver,
    ServiceServer,
    multi_threaded_spinner,
    ros_setup,
    shutdown_executors_and_threads,
)

_ = ros_setup  # Explicitly use the fixture to prevent pytest warnings


def test_ros2ari_connector_send_message(
    ros_setup: None, request: pytest.FixtureRequest
):
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    message_receiver = MessageReceiver(topic_name)
    executors, threads = multi_threaded_spinner([message_receiver])
    connector = ROS2ARIConnector()
    try:
        message = ROS2ARIMessage(
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


@pytest.mark.xfail(
    reason="Test expected to fail: ROS2 node discovery is asynchronous and the current implementation "
    "doesn't wait for topic discovery. "
    "TODO: Implement a proper discovery mechanism with timeout "
    "to ensure reliable topic communication."
)
def test_ros2ari_connector_receive_message(
    ros_setup: None, request: pytest.FixtureRequest
):
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    message_publisher = MessagePublisher(topic_name)
    executors, threads = multi_threaded_spinner([message_publisher])
    connector = ROS2ARIConnector()
    try:
        message = connector.receive_message(topic_name, timeout_sec=1.0)
        assert message.payload == String(data="Hello, ROS2!")
    finally:
        connector.shutdown()
        shutdown_executors_and_threads(executors, threads)


def test_ros2ari_connector_service_call(
    ros_setup: None, request: pytest.FixtureRequest
):
    service_name = f"{request.node.originalname}_service"  # type: ignore
    message_receiver = ServiceServer(service_name)
    executors, threads = multi_threaded_spinner([message_receiver])
    connector = ROS2ARIConnector()
    try:
        message = ROS2ARIMessage(payload={"data": True})
        response = connector.service_call(
            message, target=service_name, msg_type="std_srvs/srv/SetBool"
        )
        assert response.payload == SetBool.Response(
            success=True, message="Test service called"
        )
    finally:
        connector.shutdown()
        shutdown_executors_and_threads(executors, threads)


def test_ros2ari_connector_send_goal(ros_setup: None, request: pytest.FixtureRequest):
    action_name = f"{request.node.originalname}_action"  # type: ignore
    action_server = ActionServer(action_name)
    executors, threads = multi_threaded_spinner([action_server])
    connector = ROS2ARIConnector()
    try:
        message = ROS2ARIMessage(
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


def test_ros2ari_connector_send_goal_and_terminate_action(
    ros_setup: None, request: pytest.FixtureRequest
):
    action_name = f"{request.node.originalname}_action"  # type: ignore
    action_server = ActionServer(action_name)
    executors, threads = multi_threaded_spinner([action_server])
    connector = ROS2ARIConnector()
    feedbacks: List[Any] = []
    try:
        message = ROS2ARIMessage(payload={})
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


def test_ros2ari_connector_send_goal_erronous_callback(
    ros_setup: None, request: pytest.FixtureRequest
):
    action_name = f"{request.node.originalname}_action"  # type: ignore
    action_server = ActionServer(action_name)
    executors, threads = multi_threaded_spinner([action_server])
    connector = ROS2ARIConnector()
    try:
        message = ROS2ARIMessage(payload={})
        handle = connector.start_action(
            action_data=message,
            target=action_name,
            on_feedback=lambda feedback: 1 / 0,
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
    connector = ROS2HRIConnector(targets=[topic_name])
    hri_message_receiver = HRIMessageSubscriber(topic_name)
    executors, threads = multi_threaded_spinner([hri_message_receiver])

    try:
        images = [Image.new("RGB", (100, 100), color="red")]
        audios = [AudioSegment.silent(duration=1000)]
        text = "Hello, HRI!"
        payload = HRIPayload(images=images, audios=audios, text=text)
        message = ROS2HRIMessage(payload=payload, message_author="ai")
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
