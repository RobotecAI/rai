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
import subprocess
import typing
import unittest
from pathlib import Path
from typing import List, Optional, Tuple, get_args, get_origin
from unittest.mock import MagicMock, patch

import rclpy
from geometry_msgs.msg import TransformStamped as ROS2TransformStamped
from launch import LaunchDescription
from rai.communication.ros2 import ROS2Connector, ROS2Message
from rai.types import (
    Header,
    Point,
    Pose,
    PoseStamped,
    Quaternion,
)
from rclpy.node import Node
from rclpy.qos import QoSProfile

from rai_sim.launch_manager import ROS2LaunchManager
from rai_sim.o3de.o3de_bridge import O3DExROS2Bridge, O3DExROS2SimulationConfig
from rai_sim.simulation_bridge import Entity, SceneConfig, SpawnedEntity


def test_load_config(sample_o3dexros2_config: Path):
    config = O3DExROS2SimulationConfig.load_config(sample_o3dexros2_config)
    assert isinstance(config, O3DExROS2SimulationConfig)
    assert config.binary_path == Path("/path/to/binary")
    assert config.required_simulation_ros2_interfaces == {
        "services": ["/spawn_entity", "/delete_entity"],
        "topics": ["/color_image5", "/depth_image5", "/color_camera_info5"],
        "actions": [],
    }
    assert config.required_robotic_ros2_interfaces == {
        "services": [
            "/detection",
            "/segmentation",
            "/manipulator_move_to",
        ],
        "topics": [],
        "actions": ["/execute_trajectory"],
    }


def test_load_scene_config(sample_base_yaml_config: Path):
    config = SceneConfig.load_base_config(sample_base_yaml_config)
    assert isinstance(config.entities, list)
    assert all(isinstance(e, Entity) for e in config.entities)

    assert len(config.entities) == 2


class TestO3DExROS2Bridge(unittest.TestCase):
    @patch("rai_sim.o3de.o3de_bridge.ROS2LaunchManager")
    def setUp(self, mock_launch_manager_class):
        self.mock_connector = MagicMock(spec=ROS2Connector)
        self.mock_logger = MagicMock()

        self.mock_launch_manager = MagicMock(spec=ROS2LaunchManager)
        mock_launch_manager_class.return_value = self.mock_launch_manager

        self.bridge = O3DExROS2Bridge(
            connector=self.mock_connector, logger=self.mock_logger
        )

        # Create test data
        self.test_entity = Entity(
            name="test_entity1",
            prefab_name="cube",
            pose=PoseStamped(
                header=Header(frame_id="odom"),
                pose=Pose(
                    position=Point(x=1.0, y=2.0, z=3.0),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            ),
        )

        self.test_spawned_entity = SpawnedEntity(
            id="entity_id_123",
            name="test_entity1",
            prefab_name="cube",
            pose=PoseStamped(
                header=Header(frame_id="odom"),
                pose=Pose(
                    position=Point(x=1.0, y=2.0, z=3.0),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            ),
        )

        self.test_config = O3DExROS2SimulationConfig(
            binary_path=Path("/path/to/binary"),
            required_simulation_ros2_interfaces={
                "services": [],
                "topics": [],
                "actions": [],
            },
            required_robotic_ros2_interfaces={
                "services": [],
                "topics": [],
                "actions": [],
            },
        )

    def test_init(self):
        self.assertEqual(self.bridge.connector, self.mock_connector)
        self.assertEqual(self.bridge.logger, self.mock_logger)
        self.assertIsNone(self.bridge.current_sim_process)
        self.assertIsNone(self.bridge.current_binary_path)
        self.assertEqual(self.bridge.spawned_entities, [])

    def test_launch_robotic_stack(self):
        mock_launch_description = MagicMock(spec=LaunchDescription)

        required_interfaces = {
            "services": ["/test_service"],
            "topics": ["/test_topic"],
            "actions": ["/test_action"],
        }

        self.bridge._is_ros2_stack_ready = MagicMock(return_value=True)
        self.bridge.launch_robotic_stack(required_interfaces, mock_launch_description)

        self.mock_launch_manager.start.assert_called_once_with(
            launch_description=mock_launch_description
        )
        self.bridge._is_ros2_stack_ready.assert_called_once_with(
            required_ros2_stack=required_interfaces
        )

        self.bridge._is_ros2_stack_ready.return_value = False

        with self.assertRaises(RuntimeError):
            self.bridge.launch_robotic_stack(
                required_interfaces, mock_launch_description
            )

    @patch("subprocess.Popen")
    def test_launch_binary(self, mock_popen):
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 54322
        mock_popen.return_value = mock_process

        self.bridge._has_process_started = MagicMock(return_value=True)
        self.bridge._is_ros2_stack_ready = MagicMock(return_value=True)

        self.bridge._launch_binary(self.test_config)

        mock_popen.assert_called_once_with(["/path/to/binary"])

        self.assertEqual(self.bridge.current_sim_process, mock_process)

        self.bridge._has_process_started.assert_called_once_with(process=mock_process)
        self.bridge._is_ros2_stack_ready.assert_called_once()

    def test_shutdown_process(self):
        mock_process = MagicMock(spec=subprocess.Popen)
        mock_process.pid = 12345
        process_name = "test_process"

        mock_process.wait.return_value = 0

        self.bridge._shutdown_process(mock_process, process_name)

        mock_process.send_signal.assert_called_once_with(signal.SIGINT)
        mock_process.wait.assert_called_once()

        mock_process.reset_mock()

        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="test", timeout=15),  # SIGINT times out
            0,  # SIGTERM succeeds
        ]

        self.bridge._shutdown_process(mock_process, process_name)

        expected_calls = [
            unittest.mock.call(signal.SIGINT),
            unittest.mock.call(signal.SIGTERM),
        ]
        self.assertEqual(mock_process.send_signal.call_args_list, expected_calls)
        self.assertEqual(mock_process.wait.call_count, 2)

        mock_process.reset_mock()

        # Test case where both SIGINT and SIGTERM time out, requiring SIGKILL
        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="test", timeout=15),  # SIGINT times out
            subprocess.TimeoutExpired(cmd="test", timeout=15),  # SIGTERM times out
            0,  # SIGKILL succeeds
        ]

        self.bridge._shutdown_process(mock_process, process_name)

        expected_calls = [
            unittest.mock.call(signal.SIGINT),
            unittest.mock.call(signal.SIGTERM),
        ]
        self.assertEqual(mock_process.send_signal.call_args_list, expected_calls)
        self.assertEqual(mock_process.kill.call_count, 1)
        self.assertEqual(mock_process.wait.call_count, 3)

        # Test case where process is None
        self.bridge._shutdown_process(None, "nonexistent_process")
        # No exceptions should be raised

    def test_shutdown_binary(self):
        # Setup
        mock_process = MagicMock(spec=subprocess.Popen)
        self.bridge.current_sim_process = mock_process

        # Mock _shutdown_process
        self.bridge._shutdown_process = MagicMock()

        # Call the method
        self.bridge._shutdown_binary()

        # Verify _shutdown_process was called with the right parameters
        self.bridge._shutdown_process.assert_called_once_with(
            process=mock_process, process_name="binary"
        )

        # Verify current_sim_process was set to None
        self.assertIsNone(self.bridge.current_sim_process)

    def test_shutdown_robotic_stack(self):
        # Call the method
        self.bridge._shutdown_robotic_stack()

        # Verify manager.shutdown was called
        self.mock_launch_manager.shutdown.assert_called_once()

    def test_shutdown(self):
        # Mock the component shutdown methods
        self.bridge._shutdown_binary = MagicMock()
        self.bridge._shutdown_robotic_stack = MagicMock()

        # Call the method
        self.bridge.shutdown()

        # Verify component shutdown methods were called
        self.bridge._shutdown_binary.assert_called_once()
        self.bridge._shutdown_robotic_stack.assert_called_once()

    def test_get_available_spawnable_names(self):
        # Mock the response
        response = MagicMock()
        response.payload.model_names = ["cube", "carrot"]
        self.bridge._try_service_call = MagicMock(return_value=response)

        names = self.bridge.get_available_spawnable_names()

        self.bridge._try_service_call.assert_called_once()
        self.assertEqual(names, ["cube", "carrot"])


class TestROS2ConnectorInterface(unittest.TestCase):
    """Tests to ensure the ROS2Connector interface meets the expectations of O3DExROS2Bridge."""

    def setUp(self):
        rclpy.init()
        self.connector = ROS2Connector()

    def tearDown(self):
        rclpy.shutdown()

    def test_connector_required_methods_exist(self):
        """Test that all required methods exist on the ROS2Connector."""
        connector = ROS2Connector()

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
        self.assertTrue(
            hasattr(connector, "get_topics_names_and_types"),
            "get_topics_names_and_types method is missing",
        )
        self.assertTrue(
            hasattr(connector, "node"),
            "node property is missing",
        )

    def resolve_annotation(self, annotation: type) -> type:
        """Helper function to unwrap Optional types. Workaround for problem with asserting Optional types."""
        if get_origin(annotation) is typing.Optional:
            return get_args(annotation)[0]
        return annotation

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
            self.assertEqual(
                self.resolve_annotation(param.annotation),
                self.resolve_annotation(expected_type),
                f"Parameter '{param_name}' has incorrect type, expected: {expected_type}, got: {param.annotation}",
            )

        # Check return type explicitly
        assert signature.return_annotation is ROS2TransformStamped, (
            f"Return type is incorrect, expected: TransformStamped, got: {signature.return_annotation}"
        )

    def test_send_message_signature(self):
        signature = inspect.signature(self.connector.send_message)
        parameters = signature.parameters

        expected_params: dict[str, type] = {
            "message": ROS2Message,
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

    def test_receive_message_signature(self):
        signature = inspect.signature(self.connector.receive_message)
        parameters = signature.parameters

        expected_params: dict[str, type] = {
            "source": str,
            "timeout_sec": float,
            "msg_type": Optional[str],
            "qos_profile": Optional[QoSProfile],
            "auto_qos_matching": bool,
        }

        self.assertListEqual(
            list(parameters.keys())[: len(expected_params)],
            list(expected_params.keys()),
            f"Parameter names do not match, expected: {list(expected_params.keys())}, got: {list(parameters.keys())}",
        )

        for param_name, expected_type in expected_params.items():
            param = parameters[param_name]
            self.assertEqual(
                self.resolve_annotation(param.annotation),
                self.resolve_annotation(expected_type),
                f"Parameter '{param_name}' has incorrect type, expected: {expected_type}, got: {param.annotation}",
            )

    def test_get_topics_names_and_types_signature(self):
        signature = inspect.signature(self.connector.get_topics_names_and_types)
        parameters = signature.parameters

        expected_params: dict[str, type] = {}

        assert list(parameters.keys()) == list(expected_params.keys()), (
            f"Parameter names do not match, expected: {list(expected_params.keys())}, got: {list(parameters.keys())}"
        )

        for param_name, expected_type in expected_params.items():
            param = parameters[param_name]
            self.assertEqual(
                self.resolve_annotation(param.annotation),
                self.resolve_annotation(expected_type),
                f"Parameter '{param_name}' has incorrect type, expected: {expected_type}, got: {param.annotation}",
            )

        self.assertEqual(
            signature.return_annotation,
            List[Tuple[str, List[str]]],
            f"Return type is incorrect, expected: List[Tuple[str, List[str]]], got: {signature.return_annotation}",
        )

    def test_node_property(self):
        """Test that the node property returns the expected Node instance."""
        mock_node = MagicMock(spec=Node)
        self.connector._node = mock_node

        self.assertEqual(self.connector.node, mock_node)
        self.assertIsInstance(self.connector.node, Node)
