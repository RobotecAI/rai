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

import rclpy
import yaml
from geometry_msgs.msg import Point, Pose, Quaternion
from pydantic import BaseModel, Field, field_validator
from tf2_geometry_msgs import do_transform_pose

from rai.communication.ros2.connectors import ROS2ARIConnector, ROS2ARIMessage


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
    def __init__(self, connector: ROS2ARIConnector):
        self.connector = connector
        self.entity_ids = {}

        self.current_process = None
        self.current_binary_path = None

    def shutdown(self):
        if self.current_process:
            self.current_process.terminate()
            self.current_process = None

    def __del__(self):
        self.shutdown()

    def get_available_spawnable_names(self) -> list[str]:
        msg = ROS2ARIMessage({})

        response = self.connector.service_call(
            msg,
            target="get_available_spawnable_names",
            msg_type="gazebo_msgs/srv/GetWorldProperties",
        )

        return response.payload.model_names

    def _spawn_entity(self, entity: Entity):
        pose = do_transform_pose(
            entity.pose, self.connector.get_transform("odom", "world")
        )

        msg_content = {
            "name": entity.prefab_name,
            "xml": "",
            "robot_namespace": entity.name,
            "initial_pose": {
                "position": {
                    "x": pose.position.x,
                    "y": pose.position.y,
                    "z": pose.position.z,
                },
                "orientation": {
                    "x": pose.orientation.x,
                    "y": pose.orientation.y,
                    "z": pose.orientation.z,
                    "w": pose.orientation.w,
                },
            },
        }

        msg = ROS2ARIMessage(payload=msg_content)

        response = self.connector.service_call(
            msg, target="spawn_entity", msg_type="gazebo_msgs/srv/SpawnEntity"
        )

        if not response.payload.success:
            raise RuntimeError(response.payload.status_message)

        self.entity_ids[entity.name] = response.payload.status_message

    def _despawn_entity(self, entity: Entity):
        self._despawn_entity_by_id(self.entity_ids[entity.name])

    def _despawn_entity_by_id(self, entity_id: str):
        msg_content = {"name": entity_id}

        msg = ROS2ARIMessage(payload=msg_content)

        response = self.connector.service_call(
            msg, target="delete_entity", msg_type="gazebo_msgs/srv/DeleteEntity"
        )

        if not response.payload.success:
            raise RuntimeError(response.payload.status_message)

    def get_object_position(self, object_name: str) -> Pose:
        object_frame = object_name + "/"
        pose = do_transform_pose(
            Pose(), self.connector.get_transform(object_frame + "odom", object_frame)
        )
        pose = do_transform_pose(pose, self.connector.get_transform("world", "odom"))
        return pose

    def setup_scene(self, scene_config: SceneConfig) -> SceneSetup:
        if self.current_binary_path != scene_config.binary_path:
            if self.current_process:
                self.current_process.terminate()
            self.current_process = subprocess.Popen(
                [scene_config.binary_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.current_binary_path = scene_config.binary_path
        else:
            for entity in self.entity_ids:
                self._despawn_entity_by_id(self.entity_ids[entity])
        self.entity_ids = {}
        time.sleep(3)
        for entity in scene_config.entities:
            self._spawn_entity(entity)
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
    rclpy.init()
    connector = ROS2ARIConnector()
    o3de = O3DEEngineConnector(connector)

    scene_config = load_config(
        "src/rai_simulations/rai_simulations/example_scene1.yaml"
    )
    o3de.setup_scene(scene_config)

    import time

    time.sleep(3)

    scene_config = load_config(
        "src/rai_simulations/rai_simulations/example_scene2.yaml"
    )
    o3de.setup_scene(scene_config)

    time.sleep(3)

    o3de.shutdown()
    connector.shutdown()
    rclpy.shutdown()
