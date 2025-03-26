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
from typing import Any, Dict, List, Optional, Set

import yaml
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from geometry_msgs.msg import Pose as ROS2Pose
from rai.communication.ros2 import ROS2ARIConnector, ROS2ARIMessage
from rai.utils.ros_async import get_future_result
from std_msgs.msg import Header
from tf2_geometry_msgs import do_transform_pose

from rai_interfaces.srv import ManipulatorMoveTo
from rai_sim.simulation_bridge import (
    Entity,
    Pose,
    Rotation,
    SceneState,
    SimulationBridge,
    SimulationConfig,
    SpawnedEntity,
    Translation,
)


class O3DExROS2SimulationConfig(SimulationConfig):
    binary_path: Path
    level: Optional[str] = None
    robotic_stack_command: str
    required_simulation_ros2_interfaces: dict[str, List[str]]
    required_robotic_ros2_interfaces: dict[str, List[str]]

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
        if response.payload.success:
            return response.payload.model_names
        else:
            raise RuntimeError(
                f"Failed to get available spawnable names. Response: {response.payload.status_message}"
            )

    def _spawn_entity(self, entity: Entity):
        pose = do_transform_pose(
            self._to_ros2_pose(entity.pose),
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

    def get_object_pose(self, entity: SpawnedEntity) -> Pose:
        object_frame = entity.name + "/"
        ros2_pose = do_transform_pose(
            ROS2Pose(),
            self.connector.get_transform(object_frame + "odom", object_frame),
        )
        ros2_pose = do_transform_pose(
            ros2_pose, self.connector.get_transform("world", "odom")
        )
        return self._from_ros2_pose(ros2_pose)

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

    def _is_ros2_stack_ready(
        self, required_ros2_stack: dict[str, List[str]], retries: int = 360
    ) -> bool:
        for i in range(retries):
            available_topics = self.connector.get_topics_names_and_types()
            available_services = self.connector.node.get_service_names_and_types()
            available_topics_names = [tp[0] for tp in available_topics]
            available_services_names = [srv[0] for srv in available_services]

            # Extract action names
            available_actions_names: Set[str] = set()
            for service in available_services_names:
                if "/_action" in service:
                    action_name = service.split("/_action")[0]
                    available_actions_names.add(action_name)

            required_services = required_ros2_stack["services"]
            required_topics = required_ros2_stack["topics"]
            required_actions = required_ros2_stack["actions"]
            self.logger.info(f"required services: {required_services}")
            self.logger.info(f"required topics: {required_topics}")
            self.logger.info(f"required actions: {required_actions}")
            self.logger.info(f"available actions: {available_actions_names}")

            missing_services = [
                service
                for service in required_services
                if service not in available_services_names
            ]
            missing_topics = [
                topic
                for topic in required_topics
                if topic not in available_topics_names
            ]
            missing_actions = [
                action
                for action in required_actions
                if action not in available_actions_names
            ]

            if missing_services:
                self.logger.warning(
                    f"Waiting for missing services {missing_services} out of required services: {required_services}"
                )

            if missing_topics:
                self.logger.warning(
                    f"Waiting for missing topics: {missing_topics} out of required topics: {required_topics}"
                )

            if missing_actions:
                self.logger.warning(
                    f"Waiting for missing actions: {missing_actions} out of required actions: {required_actions}"
                )

            if not (missing_services or missing_topics or missing_actions):
                self.logger.info("All required ROS2 stack components are available.")
                return True

            time.sleep(0.5)

        self.logger.error(
            "Maximum number of retries reached. Required ROS2 stack components are not fully available."
        )
        return False

    def setup_scene(
        self,
        simulation_config: O3DExROS2SimulationConfig,
    ):
        if self.current_binary_path != simulation_config.binary_path:
            if self.current_sim_process:
                self.shutdown()
            self._launch_binary(simulation_config)
            self._launch_robotic_stack(simulation_config)
            self.current_binary_path = simulation_config.binary_path

        else:
            while self.spawned_entities:
                self._despawn_entity(self.spawned_entities[0])
            self.logger.info(f"Entities after despawn: {self.spawned_entities}")

        for entity in simulation_config.entities:
            self._spawn_entity(entity)

    def _launch_binary(
        self,
        simulation_config: O3DExROS2SimulationConfig,
    ):
        command = [
            simulation_config.binary_path.as_posix(),
        ]
        if simulation_config.level:
            command.append(f"+LoadLevel {simulation_config.level}")
        self.logger.info(f"Running command: {command}")
        self.current_sim_process = subprocess.Popen(
            command,
        )
        if not self._has_process_started(process=self.current_sim_process):
            raise RuntimeError("Process did not start in time.")
        if not self._is_ros2_stack_ready(
            required_ros2_stack=simulation_config.required_simulation_ros2_interfaces
        ):
            raise RuntimeError("ROS2 stack is not ready in time.")

    def _launch_robotic_stack(self, simulation_config: O3DExROS2SimulationConfig):
        command = shlex.split(simulation_config.robotic_stack_command)
        self.logger.info(f"Running command: {command}")
        self.current_robotic_stack_process = subprocess.Popen(
            command,
        )
        if not self._has_process_started(self.current_robotic_stack_process):
            raise RuntimeError("Process did not start in time.")
        if not self._is_ros2_stack_ready(
            required_ros2_stack=simulation_config.required_robotic_ros2_interfaces
        ):
            raise RuntimeError("ROS2 stack is not ready in time.")

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
    def _to_ros2_pose(self, pose: Pose) -> ROS2Pose:
        """
        Converts pose to pose in ROS2 Pose format.
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

        ros2_pose = ROS2Pose(position=position, orientation=orientation)

        return ros2_pose

    def _from_ros2_pose(self, pose: ROS2Pose) -> Pose:
        """
        Converts ROS2Pose to Pose
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

        return Pose(translation=translation, rotation=rotation)


class O3DEngineArmManipulationBridge(O3DExROS2Bridge):
    def reset_arm(self):
        self.connector.service_call(
            ROS2ARIMessage(payload={}),
            target="/reset_manipulator",
            msg_type="std_srvs/srv/Trigger",
        )

        self.connector.node.get_logger().debug("Reset manipulator arm: DONE")

    def move_arm(
        self,
        pose: Pose,
        initial_gripper_state: bool,
        final_gripper_state: bool,
        frame_id: str,
    ):
        """Moves arm to a given position

        Args:
            pose (Pose): where to move arm
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
        request.target_pose.pose.position.y = pose.translation.y
        request.target_pose.pose.position.z = pose.translation.z

        if pose.rotation:
            request.target_pose.pose.orientation.x = pose.rotation.x
            request.target_pose.pose.orientation.y = pose.rotation.y
            request.target_pose.pose.orientation.z = pose.rotation.z
            request.target_pose.pose.orientation.w = pose.rotation.w

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
