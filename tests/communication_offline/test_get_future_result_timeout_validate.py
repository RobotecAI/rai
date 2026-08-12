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
from unittest.mock import MagicMock

import pytest

# Import module via path without ROS tooling: ros_async imports rclpy
rclpy = pytest.importorskip("rclpy")

from rai.communication.ros2.ros_async import get_future_result  # noqa: E402


@pytest.mark.parametrize("timeout", [0, -1, True, False, "1", math.nan])
def test_get_future_result_rejects_bad_timeout(timeout):
    with pytest.raises(ValueError, match="timeout_sec"):
        get_future_result(MagicMock(), timeout_sec=timeout)
