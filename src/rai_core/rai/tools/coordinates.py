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

"""Coordinate and workspace bounds validation helpers for robot tools."""

import math
from collections.abc import Sequence


def validate_finite_coordinate(value: object, *, name: str = "coordinate") -> float:
    """Validate that a coordinate value is a finite number.

    Args:
        value: Input value to validate.
        name: Name of the coordinate parameter for error reporting.

    Returns:
        float: Validated finite float value.

    Raises:
        ValueError: If value is a boolean, non-numeric, NaN, or infinite.
    """
    if value is True or value is False or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")

    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")

    return number


def validate_workspace_bounds(
    bounds_min: Sequence[float] | None,
    bounds_max: Sequence[float] | None,
) -> None:
    """Validate that workspace bounding box specifications are well-formed.

    Args:
        bounds_min: Optional minimum (x, y, z) coordinates.
        bounds_max: Optional maximum (x, y, z) coordinates.

    Raises:
        ValueError: If bounds are improperly shaped or if minimum exceeds maximum.
    """
    axes = ("x", "y", "z")

    if bounds_min is not None:
        if len(bounds_min) != 3:
            raise ValueError(
                f"bounds_min must contain exactly 3 coordinates (x, y, z), got {len(bounds_min)}"
            )
        for i, val in enumerate(bounds_min):
            validate_finite_coordinate(val, name=f"bounds_min[{axes[i]}]")

    if bounds_max is not None:
        if len(bounds_max) != 3:
            raise ValueError(
                f"bounds_max must contain exactly 3 coordinates (x, y, z), got {len(bounds_max)}"
            )
        for i, val in enumerate(bounds_max):
            validate_finite_coordinate(val, name=f"bounds_max[{axes[i]}]")

    if bounds_min is not None and bounds_max is not None:
        for i, axis in enumerate(axes):
            min_val = float(bounds_min[i])
            max_val = float(bounds_max[i])
            if min_val > max_val:
                raise ValueError(
                    f"Invalid workspace bounds: minimum {axis} ({min_val}) cannot exceed maximum {axis} ({max_val})"
                )


def validate_pose_within_bounds(
    x: object,
    y: object,
    z: object,
    yaw: object,
    bounds_min: Sequence[float] | None = None,
    bounds_max: Sequence[float] | None = None,
) -> tuple[float, float, float, float]:
    """Validate target pose coordinates and verify they fall within workspace boundaries.

    Args:
        x: The target x coordinate.
        y: The target y coordinate.
        z: The target z coordinate.
        yaw: The target orientation yaw angle.
        bounds_min: Optional minimum (x, y, z) workspace bounds.
        bounds_max: Optional maximum (x, y, z) workspace bounds.

    Returns:
        Tuple[float, float, float, float]: Validated (x, y, z, yaw) floats.

    Raises:
        ValueError: If any coordinate is non-finite, bounds are malformed,
            or target coordinates violate configured workspace boundaries.
    """
    valid_x = validate_finite_coordinate(x, name="x")
    valid_y = validate_finite_coordinate(y, name="y")
    valid_z = validate_finite_coordinate(z, name="z")
    valid_yaw = validate_finite_coordinate(yaw, name="yaw")

    validate_workspace_bounds(bounds_min, bounds_max)

    coordinates = [("x", valid_x), ("y", valid_y), ("z", valid_z)]

    if bounds_min is not None:
        for i, (axis, val) in enumerate(coordinates):
            min_val = float(bounds_min[i])
            if val < min_val:
                raise ValueError(
                    f"Target coordinate {axis}={val} is below minimum workspace boundary {min_val}. "
                    "Goal rejected for safety."
                )

    if bounds_max is not None:
        for i, (axis, val) in enumerate(coordinates):
            max_val = float(bounds_max[i])
            if val > max_val:
                raise ValueError(
                    f"Target coordinate {axis}={val} exceeds maximum workspace boundary {max_val}. "
                    "Goal rejected for safety."
                )

    return valid_x, valid_y, valid_z, valid_yaw
