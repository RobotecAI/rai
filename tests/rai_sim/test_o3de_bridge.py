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

import inspect
import signal
import typing
import unittest
from pathlib import Path
from typing import Optional, get_args, get_origin
from unittest.mock import MagicMock, patch

import rclpy
import rclpy.qos
from geometry_msgs.msg import Point, Pose, Quaternion, TransformStamped
from rai.communication.ros2.connectors import ROS2ARIConnector, ROS2ARIMessage
from rclpy.qos import QoSProfile

from rai_sim.o3de.o3de_bridge import O3DExROS2Bridge, O3DExROS2SimulationConfig
from rai_sim.simulation_bridge import (
    Entity,
    PoseModel,
    Rotation,
    SpawnedEntity,
    Translation,
)


def test_load_config(sample_base_yaml_config: Path, sample_o3dexros2_config: Path):
    config = O3DExROS2SimulationConfig.load_config(
        sample_base_yaml_config, sample_o3dexros2_config
    )
    assert isinstance(config, O3DExROS2SimulationConfig)
    assert config.binary_path == Path("/path/to/binary")
    assert config.robotic_stack_command == "ros2 launch robotic_stack.launch.py"
    assert isinstance(config.entities, list)
    assert all(isinstance(e, Entity) for e in config.entities)

    assert len(config.entities) == 2


class TestO3DExROS2Bridge(unittest.TestCase):
    def setUp(self):
        self.mock_connector = MagicMock(spec=ROS2ARIConnector)
        self.mock_logger = MagicMock()
        self.bridge = O3DExROS2Bridge(
            connector=self.mock_connector, logger=self.mock_logger
        )

        # Create test data
        self.test_entity = Entity(
            name="test_entity1",
            prefab_name="cube",
            pose=PoseModel(
                translation=Translation(x=1.0, y=2.0, z=3.0),
                rotation=Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )

        self.test_spawned_entity = SpawnedEntity(
            id="entity_id_123",
            name="test_entity1",
            prefab_name="cube",
            pose=PoseModel(
                translation=Translation(x=1.0, y=2.0, z=3.0),
                rotation=Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )

        self.test_config = O3DExROS2SimulationConfig(
            binary_path=Path("/path/to/binary"),
            robotic_stack_command="ros2 launch robot.launch.py",
            entities=[self.test_entity],
        )

    def test_init(self):
        self.assertEqual(self.bridge.connector, self.mock_connector)
        self.assertEqual(self.bridge.logger, self.mock_logger)
        self.assertIsNone(self.bridge.current_sim_process)
        self.assertIsNone(self.bridge.current_robotic_stack_process)
        self.assertIsNone(self.bridge.current_binary_path)
        self.assertEqual(self.bridge.spawned_entities, [])

    @patch("subprocess.Popen")
    def test_launch_robotic_stack(self, mock_popen):
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 54321
        mock_popen.return_value = mock_process

        command = "ros2 launch robot.launch.py"
        self.bridge._launch_robotic_stack(command)

        mock_popen.assert_called_once_with(["ros2", "launch", "robot.launch.py"])
        self.assertEqual(self.bridge.current_robotic_stack_process, mock_process)

    @patch("subprocess.Popen")
    def test_launch_binary(self, mock_popen):
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 54322
        mock_popen.return_value = mock_process

        command = Path("/path/to/binary")
        self.bridge._launch_binary(command)

        mock_popen.assert_called_once_with(["/path/to/binary"])
        self.assertEqual(self.bridge.current_sim_process, mock_process)

    def test_shutdown_binary(self):
        mock_process = MagicMock()
        mock_process.poll.return_value = 0

        self.bridge.current_sim_process = mock_process

        self.bridge._shutdown_binary()

        mock_process.send_signal.assert_called_once_with(signal.SIGINT)
        mock_process.wait.assert_called_once()

        self.assertIsNone(self.bridge.current_sim_process)

    def test_shutdown_robotic_stack(self):
        self.bridge.current_robotic_stack_process = MagicMock()
        self.bridge.current_robotic_stack_process.poll.return_value = 0

        self.bridge._shutdown_robotic_stack()

        self.bridge.current_robotic_stack_process.send_signal.assert_called_once_with(
            signal.SIGINT
        )
        self.bridge.current_robotic_stack_process.wait.assert_called_once()

    def test_get_available_spawnable_names(self):
        # Mock the response
        response = MagicMock()
        response.payload.model_names = ["cube", "carrot"]
        self.bridge._try_service_call = MagicMock(return_value=response)

        names = self.bridge.get_available_spawnable_names()

        self.bridge._try_service_call.assert_called_once()
        self.assertEqual(names, ["cube", "carrot"])

    def test_to_ros2_pose(self):
        # Create a pose
        pose = PoseModel(
            translation=Translation(x=1.0, y=2.0, z=3.0),
            rotation=Rotation(x=0.1, y=0.2, z=0.3, w=0.4),
        )

        # Convert to ROS2 pose
        ros2_pose = self.bridge._to_ros2_pose(pose)

        # Check the conversion
        self.assertEqual(ros2_pose.position.x, 1.0)
        self.assertEqual(ros2_pose.position.y, 2.0)
        self.assertEqual(ros2_pose.position.z, 3.0)
        self.assertEqual(ros2_pose.orientation.x, 0.1)
        self.assertEqual(ros2_pose.orientation.y, 0.2)
        self.assertEqual(ros2_pose.orientation.z, 0.3)
        self.assertEqual(ros2_pose.orientation.w, 0.4)

    def test_from_ros2_pose(self):
        # Create a ROS2 pose
        position = Point(x=1.0, y=2.0, z=3.0)
        orientation = Quaternion(x=0.1, y=0.2, z=0.3, w=0.4)
        ros2_pose = Pose(position=position, orientation=orientation)

        # Convert to PoseModel
        pose = self.bridge._from_ros2_pose(ros2_pose)

        # Check the conversion
        self.assertEqual(pose.translation.x, 1.0)
        self.assertEqual(pose.translation.y, 2.0)
        self.assertEqual(pose.translation.z, 3.0)
        self.assertEqual(pose.rotation.x, 0.1)
        self.assertEqual(pose.rotation.y, 0.2)
        self.assertEqual(pose.rotation.z, 0.3)
        self.assertEqual(pose.rotation.w, 0.4)


class TestROS2ARIConnectorInterface(unittest.TestCase):
    """Tests to ensure the ROS2ARIConnector interface meets the expectations of O3DExROS2Bridge."""

    def setUp(self):
        rclpy.init()
        self.connector = ROS2ARIConnector()

    def tearDown(self):
        rclpy.shutdown()

    def test_connector_required_methods_exist(self):
        """Test that all required methods exist on the ROS2ARIConnector."""
        connector = ROS2ARIConnector()

        # Check that all required methods exist
        self.assertTrue(
            hasattr(connector, "service_call"), "service_call method is missing"
        )
        self.assertTrue(
            hasattr(connector, "get_transform"), "get_transform method is missing"
        )
        self.assertTrue(
            hasattr(connector, "send_message"), "send_message method is missing"
        )
        self.assertTrue(
            hasattr(connector, "receive_message"), "receive_message method is missing"
        )
        self.assertTrue(hasattr(connector, "shutdown"), "shutdown method is missing")

    def test_get_transform_signature(self):
        signature = inspect.signature(self.connector.get_transform)
        parameters = signature.parameters

        expected_params: dict[str, type] = {
            "target_frame": str,
            "source_frame": str,
            "timeout_sec": float,
        }

        assert list(parameters.keys()) == list(expected_params.keys()), (
            f"Parameter names do not match, expected: {list(expected_params.keys())}, got: {list(parameters.keys())}"
        )

        for param_name, expected_type in expected_params.items():
            param = parameters[param_name]
            assert param.annotation is expected_type, (
                f"Parameter '{param_name}' has incorrect type annotation, expected: {expected_type}, got: {param.annotation}"
            )

        # Check return type explicitly
        assert signature.return_annotation is TransformStamped, (
            f"Return type is incorrect, expected: TransformStamped, got: {signature.return_annotation}"
        )

    def test_send_message_signature(self):
        def resolve_annotation(annotation: type):
            """Helper function to unwrap Optional types. Workaround for problem with asserting Optional[QoSProfile] type."""
            if get_origin(annotation) is typing.Optional:
                return get_args(annotation)[0]
            return annotation

        signature = inspect.signature(self.connector.send_message)
        parameters = signature.parameters

        expected_params: dict[str, type] = {
            "message": ROS2ARIMessage,
            "target": str,
            "msg_type": str,
            "auto_qos_matching": bool,
            "qos_profile": Optional[QoSProfile],
        }

        self.assertListEqual(
            list(parameters.keys())[: len(expected_params)],
            list(expected_params.keys()),
            f"Parameter names do not match, expected: {list(expected_params.keys())}, got: {list(parameters.keys())}",
        )

        for param_name, expected_type in expected_params.items():
            param = parameters[param_name]
            self.assertEqual(
                resolve_annotation(param.annotation),
                resolve_annotation(expected_type),
                f"Parameter '{param_name}' has incorrect type, expected: {expected_type}, got: {param.annotation}",
            )

        self.assertIs(
            signature.return_annotation,
            inspect.Signature.empty,
            "send_message should have no return value",
        )

    def test_receive_message_signature(self):
        signature = inspect.signature(self.connector.receive_message)
        parameters = signature.parameters

        expected_params: dict[str, type] = {
            "source": str,
            "timeout_sec": float,
            "msg_type": Optional[str],
            "auto_topic_type": bool,
        }

        self.assertListEqual(
            list(parameters.keys())[: len(expected_params)],
            list(expected_params.keys()),
            f"Parameter names do not match, expected: {list(expected_params.keys())}, got: {list(parameters.keys())}",
        )

        for param_name, expected_type in expected_params.items():
            param = parameters[param_name]
            if isinstance(expected_type, tuple):
                self.assertIn(
                    param.annotation,
                    expected_type,
                    f"Parameter '{param_name}' has incorrect type, expected one of {expected_type}, got: {param.annotation}",
                )
            else:
                self.assertIs(
                    param.annotation,
                    expected_type,
                    f"Parameter '{param_name}' has incorrect type, expected: {expected_type}, got: {param.annotation}",
                )

        self.assertIs(
            signature.return_annotation,
            ROS2ARIMessage,
            f"Return type is incorrect, expected: ROS2ARIMessage, got: {signature.return_annotation}",
        )
