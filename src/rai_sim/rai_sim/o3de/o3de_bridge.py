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
from typing import Any, Dict, Optional

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

import yaml
from geometry_msgs.msg import Point, Pose, Quaternion
from rai.communication.ros2.connectors import ROS2ARIConnector, ROS2ARIMessage
from tf2_geometry_msgs import do_transform_pose
from rai.utils.ros_async import get_future_result
from rai_interfaces.srv import ManipulatorMoveTo
from rai_sim.simulation_bridge import (
    Entity,
    PoseModel,
    Rotation,
    SceneState,
    SimulationBridge,
    SimulationConfig,
    SpawnedEntity,
    Translation,
)


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


class O3DExROS2Bridge(SimulationBridge[O3DExROS2SimulationConfig]):
    def __init__(
        self, connector: ROS2ARIConnector, logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger=logger)
        self.connector = connector
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
            self.logger.error(
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
            self.logger.error(
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
        # PR fixing the bug: https://github.com/o3de/o3de-extras/pull/828
        # TODO (mkotynia) uncomment check if response.payload.success when the bug is fixed and remove workaround check if response.payload.model_names.

        # if response.payload.success:
        #     return response.payload.model_names
        if response.payload.model_names:
            return response.payload.model_names
        else:
            raise RuntimeError(
                f"Failed to get available spawnable names. Response: {response.payload.status_message}"
            )

    def _spawn_entity(self, entity: Entity):
        pose = do_transform_pose(
            self.to_ros2_pose(entity.pose),
            self.connector.get_transform("odom", "world"),
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
        self.logger.info(f"ros2 pose: {ros2_pose}")
        return self.from_ros2_pose(ros2_pose)

    def get_scene_state(self) -> SceneState:
        """
        Get the current scene state.
        """
        if not self.current_sim_process:
            raise RuntimeError("Simulation is not running.")
        entities: list[SpawnedEntity] = []
        for entity in self.spawned_entities:
            current_pose = self.get_object_pose(entity)
            self.logger.info(f"current pose: {current_pose}")
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
        self.logger.info(f"Running command: {command}")
        self.current_sim_process = subprocess.Popen(
            command,
        )
        if not self._has_process_started(process=self.current_sim_process):
            raise RuntimeError("Process did not start in time.")

    def _launch_robotic_stack(self, robotic_stack_command: str):
        command = shlex.split(robotic_stack_command)
        self.logger.info(f"Running command: {command}")
        self.current_robotic_stack_process = subprocess.Popen(
            command,
        )
        if not self._has_process_started(self.current_robotic_stack_process):
            raise RuntimeError("Process did not start in time.")

    def _has_process_started(self, process: subprocess.Popen[Any], timeout: int = 15):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if process.poll() is None:
                self.logger.info(f"Process started with PID {process.pid}")
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
                error_message = f"Error while calling service {target} with msg_type {msg_type}: {e}"
                self.logger.error(error_message)
                raise RuntimeError(error_message)
            if response.payload.success:
                return response
            self.logger.warning(
                f"Retrying {target}, response success: {response.payload.success}"
            )
        return response  # type: ignore

    # NOTE (mkotynia) probably to be refactored, other bridges may also want to use pose conversion to/from ROS2 format
    def to_ros2_pose(self, pose: PoseModel) -> Pose:
        """
        Converts pose in PoseModel format to pose in ROS2 Pose format.
        """
        position = Point(
            x=pose.translation.x, y=pose.translation.y, z=pose.translation.z
        )

        if pose.rotation is not None:
            orientation = Quaternion(
                x=pose.rotation.x,
                y=pose.rotation.y,
                z=pose.rotation.z,
                w=pose.rotation.w,
            )
        else:
            orientation = Quaternion()

        ros2_pose = Pose(position=position, orientation=orientation)

        return ros2_pose

    def from_ros2_pose(self, pose: Pose) -> PoseModel:
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

        return PoseModel(translation=translation, rotation=rotation)


class O3DEngineArmManipulationBridge(O3DExROS2Bridge):
    def move_arm(
        self,
        pose: PoseModel,
        initial_gripper_state: bool,
        final_gripper_state: bool,
        frame_id: str,
    ):
        """Moves arm to a given position

        Args:
            pose (PoseModel): where to move arm
            initial_gripper_state (bool): False means closed grip, True means open grip
            final_gripper_state (bool): False means closed grip, True means open grip
            frame_id (str): reference frame
        """

        request = ManipulatorMoveTo.Request()
        request.initial_gripper_state = initial_gripper_state
        request.final_gripper_state = final_gripper_state

        request.target_pose = PoseStamped()
        request.target_pose.header = Header()
        request.target_pose.header.frame_id = frame_id

        request.target_pose.pose.position.x = pose.translation.x
        request.target_pose.pose.position.x = pose.translation.y
        request.target_pose.pose.position.z = pose.translation.z

        if pose.rotation:
            request.target_pose.pose.orientation.x = pose.rotation.x
            request.target_pose.pose.orientation.y = pose.rotation.y
            request.target_pose.pose.orientation.z = pose.rotation.z
            request.target_pose.pose.orientation.w = pose.rotation.w
        else:
            request.target_pose.pose.orientation.x = 1.0
            request.target_pose.pose.orientation.y = 0.0
            request.target_pose.pose.orientation.z = 0.0
            request.target_pose.pose.orientation.w = 0.0

        client = self.connector.node.create_client(
            ManipulatorMoveTo,
            "/manipulator_move_to",
        )
        while not client.wait_for_service(timeout_sec=5.0):
            self.connector.node.get_logger().info("Service not available, waiting...")

        self.connector.node.get_logger().info("Making request to arm manipulator...")
        future = client.call_async(request)
        result = get_future_result(future, timeout_sec=5.0)

        self.connector.node.get_logger().debug(f"Moving arm result: {result}")
