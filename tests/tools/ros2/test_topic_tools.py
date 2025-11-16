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

import pytest

try:
    import rclpy  # noqa: F401

    _ = rclpy  # noqa: F841
except ImportError:
    pytest.skip("ROS2 is not installed", allow_module_level=True)

import base64
import io
import time

from PIL import Image
from rai.communication.ros2.connectors import ROS2Connector
from rai.communication.ros2.waiters import wait_for_ros2_topics
from rai.tools.ros2 import (
    GetROS2ImageTool,
    GetROS2MessageInterfaceTool,
    GetROS2TopicsNamesAndTypesTool,
    GetROS2TransformTool,
    PublishROS2MessageTool,
    ReceiveROS2MessageTool,
)
from rclpy.node import Node

from tests.communication.ros2.helpers import (
    ClockPublisher,
    ImagePublisher,
    MessagePublisher,
    MessageSubscriber,
    TransformPublisher,
    multi_threaded_spinner,
    ros_setup,
    shutdown_executors_and_threads,
)

_ = ros_setup  # Explicitly use the fixture to prevent pytest warnings


def test_publish_message_tool(ros_setup: None, request: pytest.FixtureRequest) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    connector = ROS2Connector()
    receiver = MessageSubscriber(topic=topic_name)
    executors, threads = multi_threaded_spinner([receiver])
    tool = PublishROS2MessageTool(connector=connector)
    try:
        tool._run(  # type: ignore
            topic=topic_name,
            message={"data": "test"},
            message_type="std_msgs/msg/String",
        )
        time.sleep(0.2)
        assert len(receiver.received_messages) == 1
        assert receiver.received_messages[0].data == "test"  # type: ignore
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_publish_message_tool_with_forbidden_topic(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    connector = ROS2Connector()
    tool = PublishROS2MessageTool(connector=connector, forbidden=[topic_name])
    with pytest.raises(ValueError):
        tool._run(
            topic=topic_name,
            message={"data": "test"},
            message_type="std_msgs/msg/String",
        )


def test_publish_message_tool_with_writable_topic(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    connector = ROS2Connector()
    subscriber = MessageSubscriber(topic=topic_name)
    executors, threads = multi_threaded_spinner([subscriber])
    tool = PublishROS2MessageTool(connector=connector, writable=[topic_name])
    try:
        tool._run(
            topic=topic_name,
            message={"data": "test"},
            message_type="std_msgs/msg/String",
        )  # type: ignore
        time.sleep(0.2)
        assert len(subscriber.received_messages) == 1
        assert subscriber.received_messages[0].data == "test"  # type: ignore
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_receive_message_tool(ros_setup: None, request: pytest.FixtureRequest) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    connector = ROS2Connector()
    publisher = MessagePublisher(topic=topic_name)
    executors, threads = multi_threaded_spinner([publisher])
    tool = ReceiveROS2MessageTool(connector=connector)
    time.sleep(1.0)
    try:
        response = tool._run(topic=topic_name)  # type: ignore
        time.sleep(0.2)
        assert "Hello, ROS2!" in response
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_receive_image_tool(ros_setup: None, request: pytest.FixtureRequest) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    connector = ROS2Connector()
    publisher = ImagePublisher(topic=topic_name)
    executors, threads = multi_threaded_spinner([publisher])
    tool = GetROS2ImageTool(connector=connector)
    time.sleep(1)
    try:
        _, artifact_dict = tool._run(topic=topic_name)  # type: ignore
        images = artifact_dict["images"]
        assert len(images) == 1
        image = images[0]
        image = Image.open(io.BytesIO(base64.b64decode(image)))
        assert image.size == (100, 100)
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_receive_message_tool_with_forbidden_topic(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    connector = ROS2Connector()
    tool = ReceiveROS2MessageTool(connector=connector, forbidden=[topic_name])
    with pytest.raises(ValueError):
        tool._run(topic=topic_name)  # type: ignore


def test_receive_message_tool_with_readable_topic(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    connector = ROS2Connector()
    publisher = ImagePublisher(topic=topic_name)
    executors, threads = multi_threaded_spinner([publisher])
    tool = GetROS2ImageTool(connector=connector, readable=[topic_name])
    time.sleep(1)
    try:
        _, artifact_dict = tool._run(topic=topic_name)  # type: ignore
        images = artifact_dict["images"]
        assert len(images) == 1
        image = images[0]
        image = Image.open(io.BytesIO(base64.b64decode(image)))
        assert image.size == (100, 100)
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_topics_names_and_types_tool(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    # Create a topic to ensure there's something to discover
    topic_name = f"/{request.node.originalname}_topic"  # type: ignore
    publisher = MessagePublisher(topic=topic_name)
    executors, threads = multi_threaded_spinner([publisher])
    try:
        time.sleep(0.2)  # Give time for topic to be discovered
        connector = ROS2Connector()
        tool = GetROS2TopicsNamesAndTypesTool(connector=connector)
        response = tool._run()
        assert response != ""
        # Verify the created topic appears in response
        assert topic_name in response
        # Verify response contains expected format (topic and type)
        assert "topic:" in response
        assert "type:" in response
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_topics_names_and_types_tool_with_forbidden_topic(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    # Create a topic to test forbidden exclusion
    topic_name = f"/{request.node.originalname}_topic"  # type: ignore
    forbidden_topic = f"/{request.node.originalname}_forbidden"  # type: ignore

    publisher1 = MessagePublisher(topic=topic_name)
    publisher2 = MessagePublisher(topic=forbidden_topic)

    executors, threads = multi_threaded_spinner([publisher1, publisher2])
    try:
        connector = ROS2Connector()
        wait_for_ros2_topics(
            connector, [topic_name, forbidden_topic], time_interval=0.1
        )
        tool = GetROS2TopicsNamesAndTypesTool(
            connector=connector, forbidden=[forbidden_topic]
        )
        response = tool._run()
        assert response != ""
        # Verify allowed topic appears
        assert topic_name in response
        # Verify forbidden topic does NOT appear
        assert forbidden_topic not in response
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_topics_names_and_types_tool_with_readable_topic(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    # Create topics: some readable, some not
    readable_topic1 = f"/{request.node.originalname}_readable_topic1"  # type: ignore
    non_readable_topic = f"/{request.node.originalname}_non_readable_topic"  # type: ignore

    # Create publishers for all topics
    publisher1 = MessagePublisher(topic=readable_topic1)
    publisher2 = MessagePublisher(topic=non_readable_topic)

    # Create a logger node to ensure /rosout topic is active
    logger_node = Node("test_logger_node")
    logger_node.get_logger().info("Test log message to activate /rosout")

    executors, threads = multi_threaded_spinner([publisher1, publisher2, logger_node])
    try:
        connector = ROS2Connector()
        wait_for_ros2_topics(
            connector, [readable_topic1, non_readable_topic], time_interval=0.1
        )
        readable_topics = ["/rosout", readable_topic1]
        tool = GetROS2TopicsNamesAndTypesTool(
            connector=connector, readable=readable_topics
        )
        response = tool._run()
        assert response != ""
        # Verify readable topics appear in response
        assert "/rosout" in response
        assert readable_topic1 in response
        # Verify non-readable topic does NOT appear (whitelist behavior)
        assert non_readable_topic not in response
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_topics_names_and_types_tool_with_writable_topic(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    # Create topics: some writable, some not
    writable_topic = f"/{request.node.originalname}_writable_topic"  # type: ignore
    non_writable_topic = f"/{request.node.originalname}_non_writable_topic"  # type: ignore

    # Create publishers for all topics
    publisher1 = MessagePublisher(topic=writable_topic)
    publisher2 = MessagePublisher(topic=non_writable_topic)

    executors, threads = multi_threaded_spinner([publisher1, publisher2])
    try:
        connector = ROS2Connector()
        wait_for_ros2_topics(
            connector, [writable_topic, non_writable_topic], time_interval=0.1
        )
        tool = GetROS2TopicsNamesAndTypesTool(
            connector=connector, writable=[writable_topic]
        )
        response = tool._run()
        assert response != ""
        # Verify writable topic appears in response
        assert writable_topic in response
        # Verify non-writable topic does NOT appear (whitelist behavior)
        assert non_writable_topic not in response
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_topics_names_and_types_tool_with_readable_and_writable(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    # Create topics with different combinations
    readable_only_topic = f"/{request.node.originalname}_readable_only"  # type: ignore
    writable_only_topic = f"/{request.node.originalname}_writable_only"  # type: ignore
    readable_and_writable_topic = f"/{request.node.originalname}_readable_writable"  # type: ignore
    neither_topic = f"/{request.node.originalname}_neither"  # type: ignore

    publisher1 = MessagePublisher(topic=readable_only_topic)
    publisher2 = MessagePublisher(topic=writable_only_topic)
    publisher3 = MessagePublisher(topic=readable_and_writable_topic)
    publisher4 = MessagePublisher(topic=neither_topic)

    executors, threads = multi_threaded_spinner(
        [publisher1, publisher2, publisher3, publisher4]
    )
    try:
        connector = ROS2Connector()
        wait_for_ros2_topics(
            connector,
            [
                readable_only_topic,
                writable_only_topic,
                readable_and_writable_topic,
                neither_topic,
            ],
            time_interval=0.1,
        )
        tool = GetROS2TopicsNamesAndTypesTool(
            connector=connector,
            readable=[readable_only_topic, readable_and_writable_topic],
            writable=[writable_only_topic, readable_and_writable_topic],
        )
        response = tool._run()
        assert response != ""
        # Verify readable_and_writable topic appears (should be in first section, no header)
        assert readable_and_writable_topic in response
        # Verify readable_only topic appears in Readable topics section
        assert readable_only_topic in response
        assert "Readable topics:" in response
        # Verify writable_only topic appears in Writable topics section
        assert writable_only_topic in response
        assert "Writable topics:" in response
        # Verify neither topic does NOT appear
        assert neither_topic not in response
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_topics_names_and_types_tool_with_all_restrictions(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    # Create topics
    readable_topic = f"/{request.node.originalname}_readable"  # type: ignore
    writable_topic = f"/{request.node.originalname}_writable"  # type: ignore
    forbidden_topic = f"/{request.node.originalname}_forbidden"  # type: ignore
    allowed_topic = f"/{request.node.originalname}_allowed"  # type: ignore

    publisher1 = MessagePublisher(topic=readable_topic)
    publisher2 = MessagePublisher(topic=writable_topic)
    publisher3 = MessagePublisher(topic=forbidden_topic)
    publisher4 = MessagePublisher(topic=allowed_topic)

    executors, threads = multi_threaded_spinner(
        [publisher1, publisher2, publisher3, publisher4]
    )
    try:
        connector = ROS2Connector()
        wait_for_ros2_topics(
            connector,
            [readable_topic, writable_topic, forbidden_topic, allowed_topic],
            time_interval=0.1,
        )
        tool = GetROS2TopicsNamesAndTypesTool(
            connector=connector,
            readable=[readable_topic, allowed_topic],
            writable=[writable_topic, allowed_topic],
            forbidden=[forbidden_topic],
        )
        response = tool._run()
        assert response != ""
        # Verify allowed topic appears (both readable and writable)
        assert allowed_topic in response
        # Verify readable topic appears
        assert readable_topic in response
        # Verify writable topic appears
        assert writable_topic in response
        # Verify forbidden topic does NOT appear
        assert forbidden_topic not in response
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_topics_names_and_types_tool_empty_response_when_all_filtered(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    # Create a topic that will be filtered out
    topic_name = f"/{request.node.originalname}_topic"  # type: ignore

    publisher = MessagePublisher(topic=topic_name)
    executors, threads = multi_threaded_spinner([publisher])
    try:
        connector = ROS2Connector()
        wait_for_ros2_topics(connector, [topic_name], time_interval=0.1)
        # Set readable to a topic that doesn't exist, so all topics are filtered
        tool = GetROS2TopicsNamesAndTypesTool(
            connector=connector,
            readable=[f"/{request.node.originalname}_nonexistent"],  # type: ignore
        )
        response = tool._run()
        # Response should be empty string when all topics are filtered
        assert response == ""
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_topics_names_and_types_tool_response_section_order(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    # Create topics for each category
    readable_and_writable_topic = f"/{request.node.originalname}_both"  # type: ignore
    readable_only_topic = f"/{request.node.originalname}_readable"  # type: ignore
    writable_only_topic = f"/{request.node.originalname}_writable"  # type: ignore

    publisher1 = MessagePublisher(topic=readable_and_writable_topic)
    publisher2 = MessagePublisher(topic=readable_only_topic)
    publisher3 = MessagePublisher(topic=writable_only_topic)

    executors, threads = multi_threaded_spinner([publisher1, publisher2, publisher3])
    try:
        connector = ROS2Connector()
        wait_for_ros2_topics(
            connector,
            [readable_and_writable_topic, readable_only_topic, writable_only_topic],
            time_interval=0.1,
        )
        tool = GetROS2TopicsNamesAndTypesTool(
            connector=connector,
            readable=[readable_and_writable_topic, readable_only_topic],
            writable=[readable_and_writable_topic, writable_only_topic],
        )
        response = tool._run()
        assert response != ""

        # Verify section order: readable_and_writable first (no header), then Readable, then Writable
        readable_and_writable_pos = response.find(readable_and_writable_topic)
        readable_section_pos = response.find("Readable topics:")
        writable_section_pos = response.find("Writable topics:")

        # readable_and_writable should come before Readable topics section
        assert readable_and_writable_pos < readable_section_pos
        # Readable topics section should come before Writable topics section
        assert readable_section_pos < writable_section_pos
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_message_interface_tool(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    connector = ROS2Connector()
    tool = GetROS2MessageInterfaceTool(connector=connector)
    response = tool._run(msg_type="nav2_msgs/action/NavigateToPose")  # type: ignore
    assert "goal" in response
    assert "result" in response
    assert "feedback" in response
    response = tool._run(msg_type="std_msgs/msg/String")  # type: ignore
    assert "data" in response


def test_get_transform_tool(ros_setup: None, request: pytest.FixtureRequest) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    connector = ROS2Connector()
    publisher = TransformPublisher(topic=topic_name)
    executors, threads = multi_threaded_spinner([publisher])
    tool = GetROS2TransformTool(connector=connector)
    time.sleep(1.0)
    try:
        response = tool._run(
            target_frame=publisher.frame_id,
            source_frame=publisher.child_frame_id,
            timeout_sec=1.0,
        )  # type: ignore
        assert "translation" in response
        assert "rotation" in response
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_transform_tool_with_stale_tf_data(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    """Test that GetROS2TransformTool returns stale transform data with warning."""
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    connector = ROS2Connector()
    publisher = TransformPublisher(topic=topic_name)
    executors, threads = multi_threaded_spinner([publisher])
    tool = GetROS2TransformTool(connector=connector)

    # Let the publisher run for a bit to establish transforms
    time.sleep(1.0)
    response = tool._run(
        source_frame="map",
        target_frame="base_link",
        timeout_sec=5.0,
    )  # type: ignore
    assert "translation" in response
    assert "rotation" in response

    # stop the /tf publisher
    publisher.timer.cancel()

    # wait for some time to pass then get the transform again
    time.sleep(2.0)

    try:
        response = tool._run(
            source_frame="map",
            target_frame="base_link",
            timeout_sec=5.0,
        )  # type: ignore

        response_lower = response.lower()
        staleness_indicators = [
            "stale",
            "old",
            "outdated",
            "warning",
            "invalid",
        ]
        staleness_warning_present = any(
            indicator in response_lower for indicator in staleness_indicators
        )

        #  Stale data should be warned about
        if not staleness_warning_present:
            pytest.fail(
                f"Response: {response}. "
                "Expected behavior: Either include staleness warning in response or raise appropriate error."
            )

    finally:
        # Clean shutdown
        shutdown_executors_and_threads(executors, threads)


def test_get_transform_tool_with_stale_tf_data_sim_time(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    """Test that GetROS2TransformTool returns stale transform data with warning when using sim time."""
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    connector = ROS2Connector(use_sim_time=True)

    # Create clock publisher to provide simulation time
    clock_publisher = ClockPublisher()
    publisher = TransformPublisher(topic=topic_name, use_sim_time=True)
    executors, threads = multi_threaded_spinner([clock_publisher, publisher])
    tool = GetROS2TransformTool(connector=connector)

    # Let the publishers run for a bit to establish transforms and clock
    time.sleep(1.0)
    response = tool._run(
        source_frame="map",
        target_frame="base_link",
        timeout_sec=5.0,
    )  # type: ignore
    assert "translation" in response
    assert "rotation" in response

    # stop the /tf publisher
    publisher.timer.cancel()

    # wait for some time to pass then get the transform again
    time.sleep(2.0)

    try:
        response = tool._run(
            source_frame="map",
            target_frame="base_link",
            timeout_sec=5.0,
        )  # type: ignore

        response_lower = response.lower()
        staleness_indicators = [
            "stale",
            "old",
            "outdated",
            "warning",
            "invalid",
        ]
        staleness_warning_present = any(
            indicator in response_lower for indicator in staleness_indicators
        )

        #  Stale data should be warned about
        if not staleness_warning_present:
            pytest.fail(
                f"Response: {response}. "
                "Expected behavior: Either include staleness warning in response or raise appropriate error."
            )

    finally:
        clock_publisher.timer.cancel()
        # wait for the timer to be canceled
        time.sleep(0.1)
        shutdown_executors_and_threads(executors, threads)
