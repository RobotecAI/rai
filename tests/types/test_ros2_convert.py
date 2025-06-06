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

from typing import cast

import numpy as np
import pytest
from geometry_msgs.msg import Point as ROS2Point
from geometry_msgs.msg import Pose as ROS2Pose
from geometry_msgs.msg import Quaternion as ROS2Quaternion
from rai.types import (
    Point,
    Pose,
    Quaternion,
)
from rai.types.base import RaiBaseModel
from rai.types.ros2.convert import from_ros2_msg, to_ros2_msg


def test_to_ros2_pose():
    # Create a pose
    pose = Pose(
        position=Point(x=1.0, y=2.0, z=3.0),
        orientation=Quaternion(x=0.1, y=0.2, z=0.3, w=0.4),
    )

    # Convert to ROS2 pose
    ros2_pose = to_ros2_msg(pose)

    # Check the conversion
    assert np.isclose(ros2_pose.position.x, 1.0)
    assert np.isclose(ros2_pose.position.y, 2.0)
    assert np.isclose(ros2_pose.position.z, 3.0)
    assert np.isclose(ros2_pose.orientation.x, 0.1)
    assert np.isclose(ros2_pose.orientation.y, 0.2)
    assert np.isclose(ros2_pose.orientation.z, 0.3)
    assert np.isclose(ros2_pose.orientation.w, 0.4)


def test_to_ros2_msg_wrong_input_type():
    class MyBaseModel(RaiBaseModel):
        a: int
        b: int

    msg = MyBaseModel(a=1, b=2)

    # rai.types.Ros2BaseModel is required
    with pytest.raises(TypeError):
        to_ros2_msg(msg)


def test_from_ros2_pose():
    # Create a ROS2 pose
    position = ROS2Point(x=1.0, y=2.0, z=3.0)
    orientation = ROS2Quaternion(x=0.1, y=0.2, z=0.3, w=0.4)
    ros2_pose = ROS2Pose(position=position, orientation=orientation)

    # Convert from ROS2Pose to Pose
    pose = cast(Pose, from_ros2_msg(ros2_pose))

    # Check the conversion
    assert np.isclose(pose.position.x, 1.0)
    assert np.isclose(pose.position.y, 2.0)
    assert np.isclose(pose.position.z, 3.0)
    assert np.isclose(pose.orientation.x, 0.1)
    assert np.isclose(pose.orientation.y, 0.2)
    assert np.isclose(pose.orientation.z, 0.3)
    assert np.isclose(pose.orientation.w, 0.4)
