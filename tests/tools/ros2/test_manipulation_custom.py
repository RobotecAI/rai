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

"""Manipulation tools against a real ManipulatorMoveTo service."""

import time
import uuid
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    import rclpy  # noqa: F401
except ImportError:
    pytest.skip("ROS2 is not installed", allow_module_level=True)

try:
    from rai_interfaces.srv import ManipulatorMoveTo
except ImportError:
    pytest.skip("rai_interfaces is not installed", allow_module_level=True)

from geometry_msgs.msg import TransformStamped
from langchain_core.tools import BaseTool, ToolException
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.manipulation.custom import (
    GetObjectPositionsTool,
    MoveObjectFromToTool,
    MoveToPointTool,
    ResetArmTool,
)

from tests.communication.ros2.helpers import ros_setup

_ = ros_setup  # Explicitly use the fixture to prevent pytest warnings


class Manipulator:
    """Serves ManipulatorMoveTo; answers from `outcomes` (then True), records requests."""

    def __init__(self, service: str) -> None:
        self.requests: List[Any] = []
        self.outcomes: List[bool] = []
        self.delay = 0.0
        self.server = ROS2Connector(f"manipulator{service.replace('/', '_')}")
        self.server.create_service(service, self.handle, service_type=ManipulatorMoveTo)

    def handle(self, request: Any, response: Any) -> Any:
        time.sleep(self.delay)
        self.requests.append(request)
        response.success = self.outcomes.pop(0) if self.outcomes else True
        return response

    def position(self, i: int) -> tuple:
        p = self.requests[i].target_pose.pose.position
        return (round(p.x, 3), round(p.y, 3), round(p.z, 3))

    def gripper(self, i: int) -> tuple:
        r = self.requests[i]
        return (r.initial_gripper_state, r.final_gripper_state)


@pytest.fixture
def arm(ros_setup: None):
    service = f"/move_to_{uuid.uuid4().hex}"
    manipulator = Manipulator(service)
    connector = ROS2Connector()
    kwargs = dict(connector=connector, manipulator_frame="base_link", service=service)
    try:
        yield manipulator, kwargs
    finally:
        connector.close()
        manipulator.server.close()


@pytest.mark.parametrize(
    "task,gripper,z", [("grab", (True, False), 0.3), ("drop", (False, True), 0.35)]
)
def test_move_to_point(arm, task, gripper, z) -> None:
    manipulator, kwargs = arm
    result = MoveToPointTool(**kwargs)._run(x=0.1, y=0.2, z=0.3, task=task)
    assert "successfully positioned at coordinates (0.10, 0.20, 0.30)" in result
    assert manipulator.gripper(0) == gripper
    assert manipulator.position(0) == (0.1, 0.2, z)
    assert manipulator.requests[0].target_pose.header.frame_id == "base_link"


def test_move_to_point_calibration_and_min_z(arm) -> None:
    manipulator, kwargs = arm
    tool = MoveToPointTool(**kwargs, calibration_x=0.01, calibration_y=0.02, min_z=0.2)
    tool._run(x=0.1, y=0.2, z=0.05, task="grab")
    assert manipulator.position(0) == (0.11, 0.22, 0.2)


def test_move_to_point_failure(arm) -> None:
    manipulator, kwargs = arm
    manipulator.outcomes = [False]
    result = MoveToPointTool(**kwargs)._run(x=0.1, y=0.2, z=0.3, task="grab")
    assert result.startswith("Failed to position")


def test_move_to_point_timeout(arm) -> None:
    manipulator, kwargs = arm
    manipulator.delay = 0.5
    with pytest.raises(ToolException, match="ROS2TimeoutError"):
        MoveToPointTool(**kwargs, timeout_sec=0.1)._run(x=0, y=0, z=0.3, task="grab")


def test_service_not_available(ros_setup: None) -> None:
    connector = ROS2Connector()
    tool = MoveToPointTool(
        connector=connector, manipulator_frame="base_link", service="/nobody_home"
    )
    try:
        with pytest.raises(ToolException, match="/nobody_home"):
            tool._run(x=0.1, y=0.2, z=0.3, task="grab")
    finally:
        connector.close()


def test_move_object_from_to_picks_places_and_resets(arm) -> None:
    manipulator, kwargs = arm
    result = MoveObjectFromToTool(**kwargs)._run(
        x=0.1, y=0.2, z=0.3, x1=0.4, y1=0.5, z1=0.6
    )
    assert "successfully positioned at coordinates (0.40, 0.50, 0.60)" in result
    assert [manipulator.gripper(i) for i in range(3)] == [
        (True, False),
        (False, True),
        (True, False),
    ]
    assert manipulator.position(0) == (0.1, 0.2, 0.4)  # calibration_z = 0.1
    assert manipulator.position(2) == (0.31, 0.0, 0.59)


def test_move_object_from_to_first_move_fails(arm) -> None:
    manipulator, kwargs = arm
    manipulator.outcomes = [False]
    result = MoveObjectFromToTool(**kwargs)._run(
        x=0.1, y=0.2, z=0.3, x1=0.4, y1=0.5, z1=0.6
    )
    assert result.startswith("Failed to position end effector at coordinates (0.10")
    assert len(manipulator.requests) == 1


def test_move_object_from_to_resets_after_failed_place(arm) -> None:
    manipulator, kwargs = arm
    manipulator.outcomes = [True, False]
    result = MoveObjectFromToTool(**kwargs)._run(
        x=0.1, y=0.2, z=0.3, x1=0.4, y1=0.5, z1=0.6
    )
    assert result.startswith("Failed to position end effector at coordinates (0.40")
    assert manipulator.position(2) == (0.31, 0.0, 0.59)


@pytest.mark.parametrize(
    "success,text",
    [(True, "Arm successfully reset."), (False, "Failed to reset the arm.")],
)
def test_reset_arm(arm, success, text) -> None:
    manipulator, kwargs = arm
    manipulator.outcomes = [success]
    assert ResetArmTool(**kwargs)._run() == text
    assert manipulator.position(0) == (0.31, 0.0, 0.59)


@pytest.mark.parametrize(
    "points,expected",
    [
        ([], "No cups detected."),
        (
            [([0.5, 0.0, 0.1],), ([0.1, 0.2, 0.3],)],
            "Centroids of detected cups in base_link frame: "
            "[Centroid(x=1.10, y=0.20, z=0.30), Centroid(x=1.50, y=0.00, z=0.10)]",
        ),
    ],
)
def test_get_object_positions(points, expected) -> None:
    connector = MagicMock(spec=ROS2Connector)
    transform = TransformStamped()
    transform.transform.translation.x = 1.0
    transform.transform.rotation.w = 1.0
    connector.transform = AsyncMock(return_value=transform)
    grabbing = MagicMock(spec=BaseTool)
    grabbing._run.return_value = points
    tool = GetObjectPositionsTool(
        connector=connector,
        target_frame="base_link",
        source_frame="camera",
        camera_topic="/rgb",
        depth_topic="/depth",
        camera_info_topic="/info",
        get_grabbing_point_tool=grabbing,
    )
    assert tool._run(object_name="cup").startswith(expected)
    grabbing._run.assert_called_once_with(
        camera_topic="/rgb",
        depth_topic="/depth",
        camera_info_topic="/info",
        object_name="cup",
    )
