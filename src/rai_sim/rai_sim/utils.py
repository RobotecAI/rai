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


from geometry_msgs.msg import Pose

from rai_sim.simulation_connector import PoseModel, Rotation, Translation


def ros2_pose_to_pose_model(pose: Pose) -> PoseModel:
    """
    Converts poses in ROS2 Pose format back to PoseModel format.
    """

    translation = Translation(
        x=pose.position.x,  # type: ignore
        y=pose.position.y,  # type: ignore
        z=pose.position.z,  # type: ignore
    )

    rotation = Rotation(
        x=pose.orientation.x,  # type: ignore
        y=pose.orientation.y,  # type: ignore
        z=pose.orientation.z,  # type: ignore
        w=pose.orientation.w,  # type: ignore
    )

    return PoseModel(translation=translation, rotation=rotation)
