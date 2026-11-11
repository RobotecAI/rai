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
from rai.communication.ros2.images import numpy_to_imgmsg
from rai.tools import Effect
from rai_perception.tools.segmentation_tools import (
    GetGrabbingPointTool,
    GetSegmentationTool,
)
from sensor_msgs.msg import CameraInfo

from rai_interfaces.srv import RAIGroundedSam, RAIGroundingDino

MASK = np.zeros((100, 100), dtype=np.uint8)
MASK[20:80, 30:70] = 255


@pytest.fixture
def connector(mock_connector):
    """Serves camera, depth (1 m in mm) and camera info, a detection and one mask."""
    messages = {
        "/camera": numpy_to_imgmsg(np.zeros((100, 100, 3), np.uint8), "rgb8"),
        "/depth": numpy_to_imgmsg(np.full((100, 100), 1000, np.uint16), "16UC1"),
        "/info": CameraInfo(k=[500.0, 0.0, 50.0, 0.0, 500.0, 50.0, 0.0, 0.0, 1.0]),
    }
    mock_connector.receive.side_effect = lambda topic, **_: messages[topic]

    def call(service, request, **_):
        if isinstance(request, RAIGroundingDino.Request):
            return RAIGroundingDino.Response()
        return RAIGroundedSam.Response(masks=[numpy_to_imgmsg(MASK, "mono8")])

    mock_connector.call.side_effect = call
    return mock_connector


def test_segmentation_tools_read_through_both_services(connector):
    tool = GetSegmentationTool(connector=connector, segmentation_service_name="/seg")
    assert tool.effect is Effect.READ
    assert tool.targets == ("/detection", "/seg")


def test_get_segmentation_tool_returns_masks_as_artifact(connector):
    tool = GetSegmentationTool(connector=connector)
    content, artifact = asyncio.run(
        tool._arun(camera_topic="/camera", object_name="cup")
    )
    assert content == "" and len(artifact["segmentations"]) == 1
    services = [c.args[0] for c in connector.call.call_args_list]
    assert services == ["/detection", "/segmentation"]


def test_get_grabbing_point_tool(connector):
    tool = GetGrabbingPointTool(connector=connector)
    [(centroid, rotation)] = tool._run(
        camera_topic="/camera",
        depth_topic="/depth",
        camera_info_topic="/info",
        object_name="cup",
    )
    np.testing.assert_allclose(centroid, [0.0, 0.0, 1.0], atol=0.01)
    assert -90 <= rotation <= 90
    camera_info_call = connector.receive.call_args_list[2]
    assert camera_info_call.kwargs["msg_type"] is CameraInfo


def test_process_mask_scales_depth(connector):
    tool = GetGrabbingPointTool(connector=connector, conversion_ratio=0.002)
    centroid, _ = tool._process_mask(
        numpy_to_imgmsg(MASK, "mono8"),
        numpy_to_imgmsg(np.full((100, 100), 1000, np.uint16), "16UC1"),
        (500.0, 500.0, 50.0, 50.0),
        tool.conversion_ratio,
    )
    assert centroid[2] == pytest.approx(2.0)
