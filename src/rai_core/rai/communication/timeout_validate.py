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

"""Shared guards for communication timeout parameters (no ROS deps)."""

from __future__ import annotations

from typing import Union

Number = Union[int, float]


def require_positive_timeout(timeout_sec: Number, *, name: str = "timeout_sec") -> float:
    """Return timeout as float if it is a finite number > 0.

    Raises
    ------
    TypeError
        If ``timeout_sec`` is a bool or non-numeric.
    ValueError
        If ``timeout_sec`` is <= 0.
    """
    if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)):
        raise TypeError(
            f"{name} must be a positive number, got {type(timeout_sec).__name__}"
        )
    if timeout_sec <= 0:
        raise ValueError(f"{name} must be positive, got {timeout_sec!r}")
    return float(timeout_sec)
