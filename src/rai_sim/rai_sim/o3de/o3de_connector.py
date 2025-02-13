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
import shlex
import signal
import subprocess
import time
from typing import Any, Dict

import psutil
from geometry_msgs.msg import Pose
from rai.communication.ros2.connectors import ROS2ARIConnector, ROS2ARIMessage
from tf2_geometry_msgs import do_transform_pose

from rai_sim.engine_connector import (
    EngineConnector,
    Entity,
    PoseModel,
    Rotation,
    SceneConfig,
    SceneSetup,
    Translation,
)

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
        parent = psutil.Process(self.current_process.pid)
        children = parent.children(recursive=True)

        # NOTE (mkotynia) terminating binary
        for child in children:
            logger.debug(f"Terminating child process {child.pid}, {child.name()}")
            child.terminate()

        _, alive = psutil.wait_procs(children, timeout=3)
        if alive:
            for child in alive:
                logger.warning(f"Force killing child process PID: {child.pid}")
                child.kill()
        # NOTE (mkotynia) terminating ros2 launch
        parent.send_signal(signal.SIGINT)
        parent.wait()

        if parent.is_running():
            logger.error(f"Parent process PID: {parent.pid} is still running.")
            raise RuntimeError(f"Failed to terminate main process PID {parent.pid}")

        self.current_process = None

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
            entity.pose.to_ros2_pose(), self.connector.get_transform("odom", "world")
        )

        msg_content: Dict[str, Any] = {
            "name": entity.prefab_name,
            "xml": "",
            "robot_namespace": entity.name,
            "initial_pose": {
                "position": {
                    "x": pose.position.x,  # type: ignore
                    "y": pose.position.y,  # type: ignore
                    "z": pose.position.z,  # type: ignore
                },
                "orientation": {
                    "x": pose.orientation.x,  # type: ignore
                    "y": pose.orientation.y,  # type: ignore
                    "z": pose.orientation.z,  # type: ignore
                    "w": pose.orientation.w,  # type: ignore
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

    def get_object_position(self, object_name: str) -> PoseModel:
        object_frame = object_name + "/"
        ros2_pose = do_transform_pose(
            Pose(), self.connector.get_transform(object_frame + "odom", object_frame)
        )
        ros2_pose = do_transform_pose(
            ros2_pose, self.connector.get_transform("world", "odom")
        )
        pose = self.ros2_pose_to_pose_model(ros2_pose)
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

    # TODO (mkotynia) move it to some common utils as it may be reused in other connectors
    def ros2_pose_to_pose_model(self, pose: Pose) -> PoseModel:
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

        pose_model = PoseModel(translation=translation, rotation=rotation)

        return pose_model

    def launch_binary(self, binary_path: str):
        # NOTE (mkotynia) ros2 launch command with binary path, to be refactored
        command = shlex.split(binary_path)
        logger.debug(f"Running command: {command}")
        self.current_process = subprocess.Popen(
            command,
        )
        if not self.has_process_started():
            raise RuntimeError("Process did not start in time.")

    def has_process_started(self, timeout: int = 5):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.current_process is not None:
                if self.current_process.poll() is None:
                    return True
            time.sleep(1)
        return False
