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

import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import yaml
from geometry_msgs.msg import Point, Pose, Quaternion
from pydantic import BaseModel, Field, field_validator


class Entity(BaseModel):
    name: str
    prefab_name: str
    pose: Any = Field(
        default=None
    )  # TODO (mk) consider whether make it mandatory or not

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

    binary_path: str
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


class O3DEEngineConnector(EngineConnector):
    def _spawn_entity(self, entity: Entity):
        # connector.service_call('spawn', entity)
        pass

    def _despawn_entity(self, entity: Entity):
        pass

    def get_object_position(self, object_name: str) -> Pose:
        pass

    def setup_scene(self, scene_config: SceneConfig) -> SceneSetup:
        process = subprocess.Popen(
            [scene_config.binary_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(5)
        process.terminate()
        # for entity in scene_config.entities:
        #     self._spawn_entity(entity)
        return SceneSetup(entities=scene_config.entities)


# TODO (mk) move to engine connector if SceneConfig will be common for all engines
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


if __name__ == "__main__":
    o3de_engine_connector = O3DEEngineConnector()
    o3de_engine_connector.setup_scene(load_config("scene_config.yaml"))
