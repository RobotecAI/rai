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
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, List, Optional, Set, cast

import yaml
from geometry_msgs.msg import Pose as ROS2Pose
from geometry_msgs.msg import PoseStamped as ROS2PoseStamped
from launch import LaunchDescription
from rai.communication.ros2 import ROS2Connector, ROS2Message
from rai.communication.ros2.ros_async import get_future_result
from rai.types import (
    Header,
    Pose,
    PoseStamped,
)
from rai.types.ros2 import from_ros2_msg, to_ros2_msg
from tf2_geometry_msgs import do_transform_pose, do_transform_pose_stamped

from rai_interfaces.srv import ManipulatorMoveTo
from rai_sim.launch_manager import ROS2LaunchManager
from rai_sim.simulation_bridge import (
    Entity,
    SceneConfig,
    SceneState,
    SimulationBridge,
    SimulationConfig,
    SpawnedEntity,
    SpawnEntityService,
)


class O3DExROS2SimulationConfig(SimulationConfig):
    binary_path: Path
    level: Optional[str] = None
    required_simulation_ros2_interfaces: dict[str, List[str]]
    required_robotic_ros2_interfaces: dict[str, List[str]]

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def load_config(cls, config_path: Path) -> "O3DExROS2SimulationConfig":
        with open(config_path) as f:
            connector_content: dict[str, Any] = yaml.safe_load(f)
        return cls(**connector_content)


class O3DExROS2Bridge(SimulationBridge[O3DExROS2SimulationConfig]):
    def __init__(
        self, connector: ROS2Connector, logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger=logger)
        self.connector = connector
        self.manager = ROS2LaunchManager()
        self.current_sim_process = None
        self.current_binary_path = None

    def init_simulation(self, simulation_config: O3DExROS2SimulationConfig):
        if self.current_binary_path != simulation_config.binary_path:
            if self.current_sim_process:
                self.shutdown()
            self._launch_binary(simulation_config)
            self.current_binary_path = simulation_config.binary_path

    def shutdown(self):
        self._shutdown_binary()
        self._shutdown_robotic_stack()

    def _shutdown_process(
        self,
        process: subprocess.Popen[bytes] | None,
        process_name: str,
        timeout: int = 15,
    ) -> None:
        """Shutdown a subprocess with escalating signals if needed.

        This function attempts to gracefully terminate a subprocess by first sending
        SIGINT, then SIGTERM, and finally SIGKILL if necessary. It waits for the
        specified timeout between each attempt.

        Args:
            process: The subprocess.Popen object to terminate. If None, function returns immediately.
            process_name: A descriptive name for the process (for logging purposes).
            timeout: Time in seconds to wait for the process to terminate after each signal.
                Default is 15 seconds.

        Returns:
            None
        """
        if not process:
            return

        # Try SIGINT with timeout
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            self.logger.warning(
                f"{process_name} PID: {process.pid} didn't terminate after {timeout}s with SIGINT, escalating to SIGTERM"
            )

        # Escalate to SIGTERM with timeout
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            self.logger.error(
                f"{process_name} PID: {process.pid} didn't terminate after {timeout}s with SIGTERM, escalating to SIGKILL"
            )

        # Last resort: SIGKILL
        process.kill()
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired as e:
            self.logger.critical(
                f"{process_name} PID: {process.pid} couldn't be killed! This should not happen."
            )
            raise e

    def _shutdown_binary(self):
        self._shutdown_process(process=self.current_sim_process, process_name="binary")
        self.current_sim_process = None

    def _shutdown_robotic_stack(self):
        self.manager.shutdown()

    def get_available_spawnable_names(self) -> list[str]:
        msg = ROS2Message(payload={})
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
        pose: ROS2PoseStamped = do_transform_pose_stamped(
            to_ros2_msg(entity.pose),
            self.connector.get_transform(entity.pose.header.frame_id, "world"),
        )

        msg_content = SpawnEntityService(
            name=entity.prefab_name,
            robot_namespace=entity.name,
            reference_frame=pose.header.frame_id,
            initial_pose=cast(Pose, from_ros2_msg(pose.pose)),
        )

        msg = ROS2Message(payload=msg_content.model_dump())
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

        msg = ROS2Message(payload=msg_content)

        response = self._try_service_call(
            msg, target="delete_entity", msg_type="gazebo_msgs/srv/DeleteEntity"
        )
        if response.payload.success:
            self.spawned_entities.remove(entity)
        else:
            raise RuntimeError(
                f"Failed to delete entity {entity.name}. Response: {response.payload.status_message}"
            )

    def get_object_pose(self, entity: SpawnedEntity) -> PoseStamped:
        object_frame = entity.name + "/"
        ros2_pose = do_transform_pose(
            ROS2Pose(),
            self.connector.get_transform(object_frame + "odom", object_frame),
        )
        ros2_pose = do_transform_pose(
            ros2_pose, self.connector.get_transform("world", "odom")
        )
        return PoseStamped(
            pose=cast(Pose, from_ros2_msg(ros2_pose)),
            header=Header(frame_id="odom"),
        )

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
        scene_config: SceneConfig,
    ):
        while self.spawned_entities:
            self._despawn_entity(self.spawned_entities[0])
        self.logger.info(f"Entities after despawn: {self.spawned_entities}")

        for entity in scene_config.entities:
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

    def launch_robotic_stack(
        self,
        required_robotic_ros2_interfaces: dict[str, List[str]],
        launch_description: LaunchDescription,
    ):
        self.manager.start(launch_description=launch_description)

        if not self._is_ros2_stack_ready(
            required_ros2_stack=required_robotic_ros2_interfaces
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
        self, msg: ROS2Message, target: str, msg_type: str, n_retries: int = 3
    ) -> ROS2Message:
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


class O3DEngineArmManipulationBridge(O3DExROS2Bridge):
    def reset_arm(self):
        self.connector.service_call(
            ROS2Message(payload={}),
            target="/reset_manipulator",
            msg_type="std_srvs/srv/Trigger",
        )

        self.connector.node.get_logger().debug("Reset manipulator arm: DONE")

    def move_arm(
        self,
        pose: PoseStamped,
        initial_gripper_state: bool,
        final_gripper_state: bool,
    ):
        """Moves arm to a given position

        Args:
            pose (PoseStamped): where to move arm
            initial_gripper_state (bool): False means closed grip, True means open grip
            final_gripper_state (bool): False means closed grip, True means open grip
        """

        request = ManipulatorMoveTo.Request()
        request.initial_gripper_state = initial_gripper_state
        request.final_gripper_state = final_gripper_state

        request.target_pose = to_ros2_msg(pose)

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
