# Copyright (C) 2024 Robotec.AI
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


from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import yaml
from geometry_msgs.msg import Point, Pose, Quaternion
from pydantic import BaseModel, field_validator


class Entity(BaseModel):
    name: str
    prefab_name: str
    pose: Any
    # TODO (mkotynia) consider whether create base model for poses instead of using Any as a workaround

    @field_validator("pose", mode="after")
    @classmethod
    def convert_to_pose(cls, value: Dict[str, Any]) -> Pose:
        """Convert a dict to a ROS `Pose` object."""

        translation = value.get("translation", {})
        rotation = value.get("rotation", {})

        return Pose(
            position=Point(**translation) if translation else Point(),
            orientation=Quaternion(**rotation) if rotation else Quaternion(),
        )


class SceneConfig(BaseModel):
    """
    Setup of scene - arrangmenet of objects, interactions, environment etc.
    """

    binary_path: Optional[str]
    entities: List[Entity]


class SceneSetup(BaseModel):
    """
    Info about entities in the scene (positions, collisions, etc.)
    """

    entities: List[Entity]


class EngineConnector(ABC):
    """
    Responsible for communication with simulation.
    """

    def __init__(self):
        pass

    @abstractmethod
    def setup_scene(self, scene_config: SceneConfig) -> SceneSetup:
        pass

    @abstractmethod
    def _spawn_entity(self, entity: Entity):
        pass

    @abstractmethod
    def _despawn_entity(self, entity: Entity):
        pass

    @abstractmethod
    def get_object_position(self, object_name: str) -> Pose:
        pass


def load_config(file_path: str) -> SceneConfig:
    """
    Load the scene configuration from a YAML file.
    """
    try:
        with open(file_path, "r") as file:
            content = yaml.safe_load(file)

        return SceneConfig(**content)

    except Exception as e:
        raise e
