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

"""Tests of the detection/segmentation calls and the ROS graph checks."""

import asyncio

from rai_perception.components.service_utils import (
    detect,
    missing,
    segment,
    service_available,
    wait_for,
)
from sensor_msgs.msg import Image

from rai_interfaces.msg import RAIDetectionArray


def test_detect_calls_the_detection_service(mock_connector):
    image = Image(width=4)
    result = asyncio.run(
        detect(mock_connector, image, "cup, box", service="/det", box_threshold=0.2)
    )
    assert result is mock_connector.call.return_value
    (service, request), kwargs = mock_connector.call.call_args
    assert service == "/det"
    assert request.source_img == image and request.classes == "cup, box"
    assert abs(request.box_threshold - 0.2) < 1e-6
    assert abs(request.text_threshold - 0.45) < 1e-6
    assert kwargs == {"timeout": 60.0}


def test_segment_calls_the_segmentation_service(mock_connector):
    detections = RAIDetectionArray(detection_classes=["cup"])
    asyncio.run(segment(mock_connector, Image(), detections, timeout=3.0))
    (service, request), kwargs = mock_connector.call.call_args
    assert service == "/segmentation"
    assert request.detections == detections
    assert kwargs == {"timeout": 3.0}


def test_missing_compares_resolved_names(mock_connector):
    mock_connector.get_services_names_and_types.return_value = [
        ("/detection", ["rai_interfaces/srv/RAIGroundingDino"])
    ]
    assert missing(mock_connector, ["detection", "/segmentation"]) == ["/segmentation"]
    assert service_available(mock_connector, "/detection")
    assert not service_available(mock_connector, "/segmentation")


def test_wait_for_returns_what_is_still_missing(mock_connector):
    mock_connector.get_topics_names_and_types.side_effect = [
        [],
        [("/a", ["std_msgs/msg/String"])],
        [("/a", ["std_msgs/msg/String"]), ("/b", ["std_msgs/msg/String"])],
    ]
    assert wait_for(mock_connector, ["/a", "/b"], "topic", timeout=5.0) == []
    mock_connector.get_topics_names_and_types.side_effect = None
    mock_connector.get_topics_names_and_types.return_value = []
    assert wait_for(mock_connector, ["/a"], "topic", timeout=0.05) == ["/a"]
