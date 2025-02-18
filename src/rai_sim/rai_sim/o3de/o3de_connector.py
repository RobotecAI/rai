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
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml
from geometry_msgs.msg import Pose
from rai.communication.ros2.connectors import ROS2ARIConnector, ROS2ARIMessage
from tf2_geometry_msgs import do_transform_pose

from rai_sim.simulation_connector import (
    Entity,
    PoseModel,
    SceneState,
    SimulationConfig,
    SimulationConnector,
    SpawnedEntity,
)
from rai_sim.utils import ros2_pose_to_pose_model

logger = logging.getLogger(__name__)


class O3DExROS2SimulationConfig(SimulationConfig):
    binary_path: Path
    robotic_stack_command: str

    @classmethod
    def load_config(
        cls, base_config_path: Path, connector_config_path: Path
    ) -> "O3DExROS2SimulationConfig":
        base_config = SimulationConfig.load_base_config(base_config_path)

        with open(connector_config_path) as f:
            connector_content: dict[str, Any] = yaml.safe_load(f)
        return cls(**base_config.model_dump(), **connector_content)


class O3DExROS2Connector(SimulationConnector[O3DExROS2SimulationConfig]):
    def __init__(self, connector: ROS2ARIConnector):
        self.connector = connector
        self.spawned_entities: List[
            SpawnedEntity
        ] = []  # list of spawned entities with their initial poses

        self.current_sim_process = None
        self.current_robotic_stack_process = None
        self.current_binary_path = None

    def shutdown(self):
        self._shutdown_binary()
        self._shutdown_robotic_stack()

    def _shutdown_binary(self):
        if not self.current_sim_process:
            return
        self.current_sim_process.send_signal(signal.SIGINT)
        self.current_sim_process.wait()

        if self.current_sim_process.poll() is None:
            logger.error(
                f"Parent process PID: {self.current_sim_process.pid} is still running."
            )
            raise RuntimeError(
                f"Failed to terminate main process PID {self.current_sim_process.pid}"
            )

        self.current_sim_process = None

    def _shutdown_robotic_stack(self):
        if not self.current_robotic_stack_process:
            return

        self.current_robotic_stack_process.send_signal(signal.SIGINT)
        self.current_robotic_stack_process.wait()

        if self.current_robotic_stack_process.poll() is None:
            logger.error(
                f"Parent process PID: {self.current_robotic_stack_process.pid} is still running."
            )
            raise RuntimeError(
                f"Failed to terminate robotic stack process PID {self.current_robotic_stack_process.pid}"
            )

    def get_available_spawnable_names(self) -> list[str]:
        msg = ROS2ARIMessage({})
        response = self._try_service_call(
            msg,
            target="get_available_spawnable_names",
            msg_type="gazebo_msgs/srv/GetWorldProperties",
        )
        # NOTE (mkotynia) There is a bug in the gazebo_msgs/srv/GetWorldProperties service - payload.success is not set to True even if the service call is successful. It was reported to Kacper Dąbrowski and he is going to fix it.
        if response.payload.success:
            return response.payload.model_names
        else:
            raise RuntimeError(
                f"Failed to get available spawnable names. Response: {response.payload.status_message}"
            )

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
        response = self._try_service_call(
            msg, target="spawn_entity", msg_type="gazebo_msgs/srv/SpawnEntity"
        )
        if response and response.payload.success:
            self.spawned_entities.append(
                SpawnedEntity(id=response.payload.status_message, **entity.model_dump())
            )
        else:
            raise RuntimeError(
                f"Failed to spawn entity {entity.name}. Response: {response.payload.status_message}"
            )

    def _despawn_entity(self, entity: SpawnedEntity):
        msg_content = {"name": entity.id}

        msg = ROS2ARIMessage(payload=msg_content)

        response = self._try_service_call(
            msg, target="delete_entity", msg_type="gazebo_msgs/srv/DeleteEntity"
        )
        if response.payload.success:
            self.spawned_entities.remove(entity)
        else:
            raise RuntimeError(
                f"Failed to delete entity {entity.name}. Response: {response.payload.status_message}"
            )

    def get_object_pose(self, entity: SpawnedEntity) -> PoseModel:
        object_frame = entity.name + "/"
        ros2_pose = do_transform_pose(
            Pose(), self.connector.get_transform(object_frame + "odom", object_frame)
        )
        ros2_pose = do_transform_pose(
            ros2_pose, self.connector.get_transform("world", "odom")
        )
        return ros2_pose_to_pose_model(ros2_pose)

    def get_scene_state(self) -> SceneState:
        """
        Get the current scene state.
        """
        if not self.current_sim_process:
            raise RuntimeError("Simulation is not running.")
        entities: list[SpawnedEntity] = []
        for entity in self.spawned_entities:
            current_pose = self.get_object_pose(entity)
            entities.append(
                SpawnedEntity(
                    id=entity.id,
                    name=entity.name,
                    prefab_name=entity.prefab_name,
                    pose=current_pose,
                )
            )
        return SceneState(entities=entities)

    def setup_scene(self, simulation_config: O3DExROS2SimulationConfig):
        if self.current_binary_path != simulation_config.binary_path:
            if self.current_sim_process:
                self.shutdown()
            self._launch_binary(simulation_config.binary_path)
            self._launch_robotic_stack(simulation_config.robotic_stack_command)
            self.current_binary_path = simulation_config.binary_path

        else:
            while self.spawned_entities:
                self._despawn_entity(self.spawned_entities[0])

        for entity in simulation_config.entities:
            self._spawn_entity(entity)

    def _launch_binary(self, binary_path: Path):
        command = [binary_path.as_posix()]
        logger.info(f"Running command: {command}")
        self.current_sim_process = subprocess.Popen(
            command,
        )
        if not self._has_process_started(process=self.current_sim_process):
            raise RuntimeError("Process did not start in time.")

    def _launch_robotic_stack(self, robotic_stack_command: str):
        command = shlex.split(robotic_stack_command)
        logger.info(f"Running command: {command}")
        self.current_robotic_stack_process = subprocess.Popen(
            command,
        )
        if not self._has_process_started(self.current_robotic_stack_process):
            raise RuntimeError("Process did not start in time.")

    def _has_process_started(self, process: subprocess.Popen[Any], timeout: int = 15):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if process.poll() is None:
                logger.info(f"Process started with PID {process.pid}")
                return True
            time.sleep(1)
        return False

    def _try_service_call(
        self, msg: ROS2ARIMessage, target: str, msg_type: str, n_retries: int = 3
    ) -> ROS2ARIMessage:
        if n_retries < 1:
            raise ValueError("Number of retries must be at least 1")
        for _ in range(n_retries):
            try:
                response = self.connector.service_call(
                    msg, target=target, msg_type=msg_type
                )
            except Exception as e:
                raise e
            if response.payload.success:
                return response
            logger.warning(
                f"Retrying {target}, response success: {response.payload.success}"
            )
        return response  # type: ignore
