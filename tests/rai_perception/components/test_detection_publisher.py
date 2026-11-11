# Copyright (C) 2025 Julia Jia
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

import asyncio

import pytest
from rai.communication.ros2 import QoS, ROS2TimeoutError
from rai_perception.components.detection_publisher import DetectionPublisher
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose

from rai_interfaces.srv import RAIGroundingDino


def _detection(class_id: str, score: float) -> Detection2D:
    result = ObjectHypothesisWithPose()
    result.hypothesis.class_id, result.hypothesis.score = class_id, score
    return Detection2D(results=[result])


@pytest.fixture
def publisher(mock_connector):
    mock_connector.params.update(detection_interval=0.0, default_class_threshold=0.3)
    return DetectionPublisher(connector=mock_connector)


def test_parameters_come_from_yaml_and_ros_overrides(mock_connector):
    mock_connector.params["dino_service"] = "/my_dino"
    publisher = DetectionPublisher(connector=mock_connector)
    assert publisher.params["dino_service"] == "/my_dino"
    # configs/*.yaml
    assert publisher.params["default_class_threshold"] == 0.35
    assert publisher.params["depth_topic"] == "/camera/camera/depth/image_rect_raw"
    subscribed = {
        c.args[0]: c.kwargs["msg_type"] for c in mock_connector.subscribe.call_args_list
    }
    assert subscribed == {
        "/camera/camera/color/image_raw": Image,
        "/camera/camera/depth/image_rect_raw": Image,
        "/camera/camera/color/camera_info": CameraInfo,
    }


def test_no_depth_topics_no_depth_subscriptions(mock_connector):
    mock_connector.params.update(depth_topic="", camera_info_topic="")
    DetectionPublisher(connector=mock_connector)
    assert mock_connector.subscribe.call_count == 1


@pytest.mark.parametrize(
    "classes, expected",
    [
        ("person, cup, bottle", {"person": 0.3, "cup": 0.3, "bottle": 0.3}),
        ("person:0.7, cup, bottle:0.4", {"person": 0.7, "cup": 0.3, "bottle": 0.4}),
        ("person:invalid, cup", {"person": 0.3, "cup": 0.3}),
        ("", {}),
    ],
)
def test_parse_detection_classes(publisher, classes, expected):
    names, thresholds = publisher._parse_detection_classes(classes)
    assert names == list(expected) and thresholds == expected


def test_image_callback_publishes_detections_above_class_thresholds(
    publisher, mock_connector
):
    publisher.params["detection_classes"] = "person:0.7, cup"
    response = RAIGroundingDino.Response()
    response.detections.detections = [
        _detection("person", 0.5),
        _detection("person", 0.9),
        _detection("cup", 0.35),
    ]
    response.detections.detection_classes = ["person", "cup"]
    mock_connector.call.return_value = response
    image = Image()
    image.header.frame_id = "camera"

    asyncio.run(publisher.image_callback(image))

    request = mock_connector.call.call_args.args[1]
    assert request.classes == "person, cup"
    assert abs(request.box_threshold - 0.3) < 1e-6  # min of box and class thresholds
    topic, published = mock_connector.publish.call_args.args
    assert topic == "/detection_array"
    assert mock_connector.publish.call_args.kwargs == {"qos": QoS.SENSOR}
    assert [d.results[0].hypothesis.score for d in published.detections] == [
        pytest.approx(0.9),
        pytest.approx(0.35),
    ]
    assert all(d.header.frame_id == "camera" for d in published.detections)
    assert publisher.last_detection_time > 0


def test_image_callback_survives_a_failing_service(publisher, mock_connector):
    mock_connector.call.side_effect = ROS2TimeoutError("not available")
    asyncio.run(publisher.image_callback(Image()))
    mock_connector.publish.assert_not_called()


def test_image_callback_is_throttled(publisher, mock_connector):
    mock_connector.call.return_value = RAIGroundingDino.Response()
    publisher.params["detection_interval"] = 60.0
    asyncio.run(publisher.image_callback(Image()))
    asyncio.run(publisher.image_callback(Image()))
    assert mock_connector.publish.call_count == 1
