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
# See the specific language governing permissions and
# limitations under the License.

import asyncio

import numpy as np
import pytest
from rai.communication.ros2 import ROS2TimeoutError
from rai.communication.ros2.images import numpy_to_imgmsg
from rai.tools import Effect
from rai_perception.tools.gdino_tools import (
    BoundingBox,
    DetectionData,
    GetDetectionTool,
    GetDistanceToObjectsTool,
)
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose

from rai_interfaces.srv import RAIGroundingDino


def _detection(x: float, y: float, size: float, class_id: str, score: float):
    detection = Detection2D()
    detection.bbox.center.position.x, detection.bbox.center.position.y = x, y
    detection.bbox.size_x = detection.bbox.size_y = size
    result = ObjectHypothesisWithPose()
    result.hypothesis.class_id, result.hypothesis.score = class_id, score
    detection.results = [result]
    return detection


def _serve(connector, *detections, depth_mm: int = 1000):
    """Serve a camera image, a depth image (mm) and a detection response."""
    depth = numpy_to_imgmsg(np.full((200, 200), depth_mm, np.uint16), "16UC1")
    messages = {"/camera": Image(), "/depth": depth}
    connector.receive.side_effect = lambda topic, **_: messages[topic]
    response = RAIGroundingDino.Response()
    response.detections.detections = list(detections)
    connector.call.return_value = response


def test_tools_are_read_tools_with_resolved_targets(mock_connector):
    tool = GetDetectionTool(connector=mock_connector, detection_service_name="det")
    assert tool.effect is Effect.READ
    assert tool.targets == ("/det",)
    assert tool.call_targets({"camera_topic": "cam"}) == ("/det", "/cam")
    assert set(tool.args) == {"camera_topic", "object_names"}


def test_get_detection_tool(mock_connector):
    _serve(mock_connector, _detection(50, 50, 40, "dinosaur", 0.9))
    tool = GetDetectionTool(connector=mock_connector, box_threshold=0.5)
    result = tool.invoke(
        {"camera_topic": "/camera", "object_names": ["dinosaur", "cup"]}
    )
    assert result == "I have detected the following items in the picture dinosaur"
    service, request = mock_connector.call.call_args.args
    assert service == "/detection" and request.classes == "dinosaur , cup"
    assert abs(request.box_threshold - 0.5) < 1e-6


def test_get_detection_tool_nothing_detected(mock_connector):
    _serve(mock_connector)
    tool = GetDetectionTool(connector=mock_connector)
    result = asyncio.run(
        tool.ainvoke({"camera_topic": "/camera", "object_names": ["x"]})
    )
    assert result.endswith("picture None")


def test_service_failure_reaches_the_llm_as_an_error(mock_connector):
    _serve(mock_connector)
    mock_connector.call.side_effect = ROS2TimeoutError(
        "Service /detection not available"
    )
    tool = GetDetectionTool(connector=mock_connector)
    result = tool.invoke({"camera_topic": "/camera", "object_names": ["x"]})
    assert "not available" in result


def test_unreadable_topic_is_refused(mock_connector):
    _serve(mock_connector)
    tool = GetDetectionTool(connector=mock_connector, forbidden=["/camera"])
    result = tool.invoke({"camera_topic": "/camera", "object_names": ["x"]})
    assert "not readable" in result
    mock_connector.receive.assert_not_called()


def test_get_distance_to_objects_tool(mock_connector):
    _serve(mock_connector, _detection(50, 50, 40, "dinosaur", 0.9), depth_mm=1500)
    tool = GetDistanceToObjectsTool(connector=mock_connector)
    result = tool.invoke(
        {"camera_topic": "/camera", "depth_topic": "/depth", "object_names": ["d"]}
    )
    assert (
        result
        == "I have detected the following items in the picture dinosaur: 1.50m away"
    )


def test_distance_from_detections_removes_outliers(mock_connector):
    depth = np.full((100, 100), 1000, np.uint16)
    depth[10, 10] = 9000  # an outlier in the box
    detection = DetectionData(
        class_name="cup",
        confidence=0.9,
        bbox=BoundingBox(x_center=20, y_center=20, width=30, height=30),
    )
    tool = GetDistanceToObjectsTool(connector=mock_connector)
    [(name, distance)] = tool._get_distance_from_detections(
        numpy_to_imgmsg(depth, "16UC1"), [detection], 1.0, 0.001
    )
    assert name == "cup" and distance == pytest.approx(1.0)


def test_check_service_dependencies(mock_connector):
    mock_connector.get_services_names_and_types.return_value = [
        ("/detection", ["rai_interfaces/srv/RAIGroundingDino"])
    ]
    tool = GetDetectionTool(connector=mock_connector)
    assert tool.check_service_dependencies() == {"/detection": True}
