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

import math

import pytest
from rai.tools.coordinates import (
    validate_finite_coordinate,
    validate_pose_within_bounds,
    validate_workspace_bounds,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, 0.0),
        (1, 1.0),
        (-5, -5.0),
        (3.14159, 3.14159),
        (-42.75, -42.75),
        (1e6, 1e6),
    ],
)
def test_validate_finite_coordinate_accepts(value, expected):
    assert validate_finite_coordinate(value, name="test_coord") == expected


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        math.nan,
        math.inf,
        -math.inf,
        "1.0",
        "invalid",
        None,
        [],
        {},
    ],
)
def test_validate_finite_coordinate_rejects(value):
    with pytest.raises(ValueError, match="finite number"):
        validate_finite_coordinate(value, name="test_coord")


def test_validate_workspace_bounds_accepts():
    validate_workspace_bounds(None, None)
    validate_workspace_bounds((-10.0, -10.0, -0.5), (10.0, 10.0, 3.0))
    validate_workspace_bounds([-5, 0, 0], [5, 10, 2])
    validate_workspace_bounds((-10.0, -10.0, 0.0), None)
    validate_workspace_bounds(None, (10.0, 10.0, 3.0))


@pytest.mark.parametrize(
    "bounds",
    [
        (1.0, 2.0),
        (1.0, 2.0, 3.0, 4.0),
        [],
    ],
)
def test_validate_workspace_bounds_rejects_length(bounds):
    with pytest.raises(ValueError, match="exactly 3 coordinates"):
        validate_workspace_bounds(bounds, None)

    with pytest.raises(ValueError, match="exactly 3 coordinates"):
        validate_workspace_bounds(None, bounds)


@pytest.mark.parametrize(
    "bounds",
    [
        (1.0, math.nan, 3.0),
        (math.inf, 2.0, 3.0),
        (1.0, 2.0, True),
        (1.0, "2.0", 3.0),
    ],
)
def test_validate_workspace_bounds_rejects_non_finite(bounds):
    with pytest.raises(ValueError, match="finite number"):
        validate_workspace_bounds(bounds, None)


@pytest.mark.parametrize(
    "min_bounds,max_bounds,violated_axis",
    [
        ((5.0, 0.0, 0.0), (2.0, 10.0, 2.0), "x"),
        ((0.0, 15.0, 0.0), (10.0, 5.0, 2.0), "y"),
        ((0.0, 0.0, 5.0), (10.0, 10.0, 1.0), "z"),
    ],
)
def test_validate_workspace_bounds_rejects_inverted(
    min_bounds, max_bounds, violated_axis
):
    with pytest.raises(
        ValueError,
        match=f"minimum {violated_axis} .* cannot exceed maximum {violated_axis}",
    ):
        validate_workspace_bounds(min_bounds, max_bounds)


def test_validate_pose_within_bounds_unbounded():
    x, y, z, yaw = validate_pose_within_bounds(1.5, -2.5, 0.25, 0.785)
    assert x == 1.5
    assert y == -2.5
    assert z == 0.25
    assert yaw == 0.785


def test_validate_pose_within_bounds_bounded_valid():
    bounds_min = (-10.0, -10.0, -0.5)
    bounds_max = (10.0, 10.0, 3.0)
    x, y, z, yaw = validate_pose_within_bounds(
        x=5.0, y=-3.0, z=1.2, yaw=1.57, bounds_min=bounds_min, bounds_max=bounds_max
    )
    assert (x, y, z, yaw) == (5.0, -3.0, 1.2, 1.57)


def test_validate_pose_within_bounds_boundary_limits():
    bounds_min = (-10.0, -5.0, 0.0)
    bounds_max = (10.0, 5.0, 2.0)
    # Exact boundaries should be allowed
    x, y, z, yaw = validate_pose_within_bounds(
        x=-10.0, y=5.0, z=0.0, yaw=0.0, bounds_min=bounds_min, bounds_max=bounds_max
    )
    assert (x, y, z, yaw) == (-10.0, 5.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "x,y,z,violated_axis,direction",
    [
        (-10.1, 0.0, 1.0, "x", "below minimum"),
        (10.1, 0.0, 1.0, "x", "exceeds maximum"),
        (0.0, -5.5, 1.0, "y", "below minimum"),
        (0.0, 5.5, 1.0, "y", "exceeds maximum"),
        (0.0, 0.0, -0.1, "z", "below minimum"),
        (0.0, 0.0, 3.1, "z", "exceeds maximum"),
    ],
)
def test_validate_pose_within_bounds_violations(x, y, z, violated_axis, direction):
    bounds_min = (-10.0, -5.0, 0.0)
    bounds_max = (10.0, 5.0, 3.0)

    pattern = f"Target coordinate {violated_axis}=.* {direction} workspace boundary"
    with pytest.raises(ValueError, match=pattern):
        validate_pose_within_bounds(
            x=x, y=y, z=z, yaw=0.0, bounds_min=bounds_min, bounds_max=bounds_max
        )


def test_validate_pose_within_bounds_rejects_non_finite_inputs():
    bounds_min = (-10.0, -10.0, 0.0)
    bounds_max = (10.0, 10.0, 3.0)

    with pytest.raises(ValueError, match="x must be a finite number"):
        validate_pose_within_bounds(math.nan, 0.0, 0.0, 0.0, bounds_min, bounds_max)

    with pytest.raises(ValueError, match="yaw must be a finite number"):
        validate_pose_within_bounds(0.0, 0.0, 0.0, math.inf, bounds_min, bounds_max)
