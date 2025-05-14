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

import logging
import math
from typing import List, Tuple, Union

from rai.types import Quaternion
from rclpy.impl.rcutils_logger import RcutilsLogger

from rai_bench.manipulation_o3de.interfaces import (
    ManipulationTask,
)
from rai_sim.simulation_bridge import Entity, SceneConfig

loggers_type = Union[RcutilsLogger, logging.Logger]


class RotateObjectTask(ManipulationTask):
    def __init__(
        self,
        obj_types: List[str],
        target_quaternion: Quaternion,
        logger: loggers_type | None = None,
    ):
        # NOTE (jmatejcz) for now manipulaiton tool does not support passing rotation
        # NOTE (jmatejcz) rotating around other axis than z seems to not have much sense as the objecet will fall
        # can the target rotation be expressed differently? maybe only by rotation around z axis as an angle?
        """
        Parameters
        ----------
        obj_types : List[str]
            List of allowed object types that will be rotated.
        target_quaternion : Tuple[float, float, float, float]
            The target rotation expressed as a quaternion (x, y, z, w).
        """
        super().__init__(logger=logger)
        self.obj_types = obj_types
        self.target_quaternion = target_quaternion

    @property
    def task_prompt(self) -> str:
        object_names = ", ".join(obj.replace("_", " ") for obj in self.obj_types)
        return (
            f"Rotate each {object_names} to the target orientation specified by the quaternion "
            f"- x:{self.target_quaternion.x}, y:{self.target_quaternion.y}, z:{self.target_quaternion.z}, w:{self.target_quaternion.w} "
            "Remember to rotate the gripper when grabbing objects."
        )

    def check_if_required_objects_present(self, simulation_config: SceneConfig) -> bool:
        """
        Validate that at least one object of the specified types is present.

        Returns
        -------
        bool
            True if at least one allowed object is present, False otherwise.
        """
        return any(
            ent.prefab_name in self.obj_types for ent in simulation_config.entities
        )

    def calculate_correct(
        self, entities: List[Entity], allowable_rotation_error: float = 5.0
    ) -> Tuple[int, int]:
        """
        Calculate the number of correctly rotated objects and incorrectly rotated objects,
        operating on quaternion representations.

        For each object, the dot product between its rotation quaternion and the target quaternion
        is computed. The angular difference is calculated as:

            angle_diff = 2 * acos(|dot(current, target)|)

        This value (converted from radians to degrees) is compared with the allowable rotation error.
        If the difference is within the allowable error, the object's orientation is considered correct.

        Parameters
        ----------
        entities : List[Entity]
            List of all entities present in the simulation scene.
        allowable_rotation_error : float, optional
            The acceptable deviation (in degrees) from the target rotation. Defaults to 5.0.

        Returns
        -------
        Tuple[int, int]
            A tuple where the first element is the number of correctly rotated objects and the second element
            is the number of incorrectly rotated objects.
        """
        correct = 0
        incorrect = 0
        for entity in entities:
            if entity.prefab_name in self.obj_types:
                if not entity.pose.pose.orientation:
                    ValueError("Entity has no rotation defined.")
                else:
                    dot = (
                        entity.pose.pose.orientation.x * self.target_quaternion.x
                        + entity.pose.pose.orientation.y * self.target_quaternion.y
                        + entity.pose.pose.orientation.z * self.target_quaternion.z
                        + entity.pose.pose.orientation.w * self.target_quaternion.w
                    )
                    # Account for the double cover: q and -q represent the same rotation.
                    dot = abs(dot)
                    # Clamp the dot product to avoid domain errors.
                    dot = max(min(dot, 1.0), -1.0)
                    angle_diff_deg = math.degrees(2 * math.acos(dot))
                    if angle_diff_deg <= allowable_rotation_error:
                        correct += 1
                    else:
                        incorrect += 1
        return correct, incorrect
