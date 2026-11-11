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

import math

import pytest
from geometry_msgs.msg import Quaternion
from rai.tools.ros2.navigation.geometry import quaternion_to_yaw, yaw_to_quaternion


@pytest.mark.parametrize("yaw", [0.0, 0.5, -1.2, math.pi / 2, 3.0, -3.0])
def test_yaw_round_trip(yaw):
    q = yaw_to_quaternion(yaw)
    assert math.isclose(q.x**2 + q.y**2 + q.z**2 + q.w**2, 1.0)
    assert math.isclose(quaternion_to_yaw(q), yaw, abs_tol=1e-12)


def test_yaw_ignores_roll_and_pitch():
    # roll = 0.3 then yaw = 0.7 (ZYX): q = q_yaw * q_roll
    cr, sr = math.cos(0.15), math.sin(0.15)
    cy, sy = math.cos(0.35), math.sin(0.35)
    q = Quaternion(x=cy * sr, y=sy * sr, z=sy * cr, w=cy * cr)
    assert math.isclose(quaternion_to_yaw(q), 0.7, abs_tol=1e-12)
