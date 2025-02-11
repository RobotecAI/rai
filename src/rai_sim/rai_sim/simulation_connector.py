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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, List, Optional, TypeVar

import yaml
from geometry_msgs.msg import Point, Pose, Quaternion
from pydantic import BaseModel, field_validator


class Translation(BaseModel):
    x: float
    y: float
    z: float


class Rotation(BaseModel):
    x: float
    y: float
    z: float
    w: float


class PoseModel(BaseModel):
    translation: Translation
    rotation: Optional[Rotation]

    def to_ros2_pose(self) -> Pose:
        """
        Converts pose in PoseModel format to pose in ROS2 Pose format.
        """

        position = Point(
            x=self.translation.x, y=self.translation.y, z=self.translation.z
        )

        if self.rotation is not None:
            orientation = Quaternion(
                x=self.rotation.x,
                y=self.rotation.y,
                z=self.rotation.z,
                w=self.rotation.w,
            )
        else:
            orientation = Quaternion()

        ros2_pose = Pose(position=position, orientation=orientation)

        return ros2_pose

    @classmethod
    def from_ros2_pose(cls, pose: Pose):
        """
        Converts ROS2 pose to PoseModel format
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

        return cls(translation=translation, rotation=rotation)


class Entity(BaseModel):
    name: str
    prefab_name: str
    pose: PoseModel


class SpawnedEntity(Entity):
    id: str


class SimulationConfig(BaseModel):
    """
    Setup of simulation - arrangemenet of objects in the environment.
    """

    # NOTE (mkotynia) can be extended by other attributes
    entities: List[Entity]

    @field_validator("entities")
    @classmethod
    def check_unique_names(cls, entities: List[Entity]) -> List[Entity]:
        names = [entity.name for entity in entities]
        if len(names) != len(set(names)):
            raise ValueError("Each entity must have a unique name.")
        return entities

    @classmethod
    def load_base_config(cls, base_config_path: Path) -> "SimulationConfig":
        with open(base_config_path) as f:
            content = yaml.safe_load(f)
        return cls(**content)


class SceneState(BaseModel):
    """
    Info about current entities' state in the scene.
    """

    # NOTE (mkotynia) can be extended by other attributes
    entities: List[SpawnedEntity]


SimulationConfigT = TypeVar("SimulationConfigT", bound=SimulationConfig)


class SimulationConnector(ABC, Generic[SimulationConfigT]):
    """
    Responsible for communication with simulation.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    @abstractmethod
    def setup_scene(self, simulation_config: SimulationConfigT):
        pass

    @abstractmethod
    def _spawn_entity(self, entity: Entity):
        pass

    @abstractmethod
    def _despawn_entity(self, entity: SpawnedEntity):
        pass

    @abstractmethod
    def get_object_pose(self, entity: SpawnedEntity) -> PoseModel:
        pass

    @abstractmethod
    def get_scene_state(self) -> SceneState:
        pass

    def shutdown(self):
        pass
