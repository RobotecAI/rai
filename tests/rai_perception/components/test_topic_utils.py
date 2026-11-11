# Copyright (C) 2025 Robotec.AI
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

from unittest.mock import Mock

import pytest
from langchain_core.tools import BaseTool
from rai.communication.ros2 import ROS2ServiceError
from rai_perception import (
    GetDetectionTool,
    GetObjectGrippingPointsTool,
    discover_camera_topics,
    wait_for_perception_dependencies,
)

TOPICS = {
    "camera_topic": "/test/camera",
    "depth_topic": "/test/depth",
    "camera_info_topic": "/test/camera_info",
}


def _graph(connector, services=(), topics=()):
    connector.get_services_names_and_types.return_value = [
        (s, ["pkg/srv/T"]) for s in services
    ]
    connector.get_topics_names_and_types.return_value = [
        (t, ["sensor_msgs/msg/Image"]) for t in topics
    ]


class TestWaitForPerceptionDependencies:
    def test_gripping_points_tool_needs_both_services_and_its_topics(
        self, mock_connector
    ):
        tool = GetObjectGrippingPointsTool(connector=mock_connector, **TOPICS)
        _graph(mock_connector, ["/detection", "/segmentation"], TOPICS.values())
        wait_for_perception_dependencies(mock_connector, [Mock(spec=BaseTool), tool])

    def test_missing_segmentation_service_raises(self, mock_connector):
        tool = GetObjectGrippingPointsTool(connector=mock_connector, **TOPICS)
        _graph(mock_connector, ["/detection"], TOPICS.values())
        with pytest.raises(ROS2ServiceError) as error:
            wait_for_perception_dependencies(mock_connector, [tool], timeout=0.1)
        assert error.value.service_name == "/segmentation"

    def test_missing_topic_raises_timeout_with_hints(self, mock_connector):
        tool = GetObjectGrippingPointsTool(connector=mock_connector, **TOPICS)
        _graph(
            mock_connector, ["/detection", "/segmentation"], ["/test/camera", "/rgb"]
        )
        with pytest.raises(TimeoutError, match="/test/depth"):
            wait_for_perception_dependencies(mock_connector, [tool], timeout=0.1)

    def test_detection_only_tool_needs_only_its_service(self, mock_connector):
        tool = GetDetectionTool(
            connector=mock_connector, detection_service_name="/my_detection"
        )
        _graph(mock_connector, ["/my_detection"])
        wait_for_perception_dependencies(mock_connector, [tool])

    @pytest.mark.parametrize("tools", [[], [Mock(spec=BaseTool)]])
    def test_without_perception_tools(self, mock_connector, tools):
        with pytest.raises(RuntimeError, match="No perception tools"):
            wait_for_perception_dependencies(mock_connector, tools)


def test_discover_camera_topics(mock_connector):
    mock_connector.get_topics_names_and_types.return_value = [
        ("/camera/color", ["sensor_msgs/msg/Image"]),
        ("/camera/depth", ["sensor_msgs/msg/Image"]),
        ("/camera/info", ["sensor_msgs/msg/CameraInfo"]),
        ("/chatter", ["std_msgs/msg/String"]),
    ]
    assert discover_camera_topics(mock_connector) == {
        "image_topics": ["/camera/color"],
        "depth_topics": ["/camera/depth"],
        "camera_info_topics": ["/camera/info"],
        "all_topics": ["/camera/color", "/camera/depth", "/camera/info", "/chatter"],
    }


def test_discover_camera_topics_when_the_graph_query_fails(mock_connector):
    mock_connector.get_topics_names_and_types.side_effect = RuntimeError("closed")
    assert discover_camera_topics(mock_connector)["all_topics"] == []
