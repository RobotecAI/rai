# Copyright (C) 2026 Robotec.AI
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

"""A navigation tool configured with workspace bounds must refuse a goal
outside them before it reaches the action server."""

import pytest

try:
    import rclpy  # noqa: F401

    _ = rclpy  # noqa: F841
except ImportError:
    pytest.skip("ROS2 is not installed", allow_module_level=True)

from unittest.mock import MagicMock

from pydantic import ValidationError
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.navigation.nav2 import Nav2Toolkit, NavigateToPoseTool
from rai.tools.ros2.navigation.nav2_blocking import NavigateToPoseBlockingTool

MIN = (-5.0, -5.0, 0.0)
MAX = (5.0, 5.0, 2.0)


@pytest.fixture
def connector():
    return MagicMock(spec=ROS2Connector)


@pytest.fixture(params=[NavigateToPoseTool, NavigateToPoseBlockingTool])
def bounded_tool(request, connector):
    return request.param(
        connector=connector, workspace_bounds_min=MIN, workspace_bounds_max=MAX
    )


@pytest.mark.parametrize(
    "x,y,z",
    [
        (6.0, 0.0, 1.0),
        (-6.0, 0.0, 1.0),
        (0.0, 6.0, 1.0),
        (0.0, 0.0, 3.0),
        (0.0, 0.0, -1.0),
    ],
)
def test_goal_outside_bounds_is_refused(bounded_tool, x, y, z):
    with pytest.raises(ValueError, match="Goal rejected"):
        bounded_tool._run(x=x, y=y, z=z, yaw=0.0)
    bounded_tool.connector.start_action.assert_not_called()


def test_unbounded_tool_accepts_anything(connector):
    NavigateToPoseTool(connector=connector).reject_out_of_bounds(1e9, -1e9, 1e9)


def test_one_sided_bounds_only_constrain_that_side(connector):
    tool = NavigateToPoseTool(connector=connector, workspace_bounds_max=MAX)
    tool.reject_out_of_bounds(-1e9, -1e9, 0.0)
    with pytest.raises(ValueError, match="outside the workspace"):
        tool.reject_out_of_bounds(6.0, 0.0, 0.0)


def test_inverted_bounds_are_rejected_at_construction(connector):
    with pytest.raises(ValidationError, match="exceeds"):
        NavigateToPoseTool(
            connector=connector, workspace_bounds_min=MAX, workspace_bounds_max=MIN
        )


def test_nan_bounds_are_rejected_at_construction(connector):
    with pytest.raises(ValidationError):
        NavigateToPoseTool(
            connector=connector, workspace_bounds_min=(1.0, 2.0, float("nan"))
        )


def test_toolkit_passes_bounds_to_the_goal_tool(connector):
    toolkit = Nav2Toolkit(
        connector=connector, workspace_bounds_min=MIN, workspace_bounds_max=MAX
    )
    tool = next(t for t in toolkit.get_tools() if isinstance(t, NavigateToPoseTool))
    assert tool.workspace_bounds_min == MIN
    assert tool.workspace_bounds_max == MAX
