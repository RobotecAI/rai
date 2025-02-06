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

import time

from rai.communication.ros2.connectors import ROS2ARIConnector
from rai.tools.ros2 import GetROS2TransformTool
from tests.communication.ros2.helpers import (
    TransformPublisher,
    multi_threaded_spinner,
    ros_setup,
    shutdown_executors_and_threads,
)

_ = ros_setup  # Explicitly use the fixture to prevent pytest warnings


def test_get_transform_tool(ros_setup: None, request: pytest.FixtureRequest) -> None:
    topic_name = f"{request.node.originalname}_topic"  # type: ignore
    connector = ROS2ARIConnector()
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
