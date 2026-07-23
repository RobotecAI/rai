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

"""Pure positive-parameter guards for rai_sim (no O3DE / ROS imports)."""

from __future__ import annotations


def require_positive_number(value, *, name: str = "value") -> float:
    """Return value as float if it is a finite number > 0 (rejects bool impostors)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a positive number, got {type(value).__name__}"
        )
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return float(value)
