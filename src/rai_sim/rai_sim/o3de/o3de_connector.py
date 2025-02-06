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

import logging
import os
import shlex
import signal
import subprocess
import time
from typing import Dict

import psutil
from geometry_msgs.msg import Pose
from rai.communication.ros2.connectors import ROS2ARIConnector, ROS2ARIMessage
from tf2_geometry_msgs import do_transform_pose

from rai_sim.engine_connector import EngineConnector, Entity, SceneConfig, SceneSetup

logger = logging.getLogger(__name__)


class O3DEngineConnector(EngineConnector):
    def __init__(self, connector: ROS2ARIConnector):
        self.connector = connector
        self.entity_ids: Dict[str, str] = {}

        self.current_process = None
        self.current_binary_path = None

    def shutdown(self):
        if not self.current_process:
            return
        try:
            logger.debug(f"Sending SIGINT to process {self.current_process.pid}")
            self.current_process.send_signal(signal.SIGINT)

            parent = psutil.Process(self.current_process.pid)
            children = parent.children(recursive=True)
            processes = children + [parent]

            _, alive = psutil.wait_procs(processes, timeout=3)

            # NOTE (mkotynia) kill ros2
            for process in alive:
                logger.debug(f"Force killing process {process.pid}, {process.name()}")
                try:
                    process.kill()
                except Exception as e:
                    logger.warning(f"Failed to kill process {process.pid}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
            try:
                if self.current_process:
                    os.killpg(os.getpgid(self.current_process.pid), signal.SIGKILL)
            except Exception as e:
                logger.debug(f"Failed to kill process group: {e}")

        finally:
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
            if scene_config.binary_path:
                self.launch_binary(scene_config.binary_path)
            else:
                raise Exception("No binary path provided")
            self.current_binary_path = scene_config.binary_path
        else:
            for entity in self.entity_ids:
                self._despawn_entity_by_id(self.entity_ids[entity])
        self.entity_ids = {}
        time.sleep(3)
        for entity in scene_config.entities:
            self._spawn_entity(entity)
        # TODO (mkotynia) handle SceneSetup
        return SceneSetup(entities=scene_config.entities)

    def launch_binary(self, binary_path: str):
        # NOTE (mkotynia) ros2 launch command with binary path, to be refactored
        command = shlex.split(binary_path)
        logger.info(f"Running command: {command}")
        self.current_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
