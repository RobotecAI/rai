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

import signal
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from launch import LaunchDescription
from rai.communication.ros2 import QoS, ROS2Connector
from rai.types import (
    Header,
    Point,
    Pose,
    PoseStamped,
    Quaternion,
)
from std_srvs.srv import SetBool

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
            "/grounding_dino_classify",
            "/grounded_sam_segment",
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
        self.mock_connector.sync = MagicMock()
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
        response.model_names = ["cube", "carrot"]
        self.bridge._call_with_retries = MagicMock(return_value=response)

        names = self.bridge.get_available_spawnable_names()

        self.bridge._call_with_retries.assert_called_once()
        self.assertEqual(names, ["cube", "carrot"])

    def test_connect_object_tf_tree_to_world(self):
        """Every static transform so far is published latched on /tf_static."""
        from geometry_msgs.msg import PoseStamped as ROS2PoseStamped

        pose_in_world = ROS2PoseStamped()
        pose_in_world.header.frame_id = "world"
        pose_in_world.pose.position.x = 1.0
        pose_in_world.pose.position.y = 2.0
        pose_in_world.pose.position.z = 3.0
        pose_in_world.pose.orientation.w = 1.0

        self.bridge._connect_object_tf_tree_to_world(
            self.test_spawned_entity, pose_in_world
        )
        other = self.test_spawned_entity.model_copy(update={"name": "other"})
        self.bridge._connect_object_tf_tree_to_world(other, pose_in_world)

        publish = self.mock_connector.sync.publish
        self.assertEqual(publish.call_count, 2)
        topic, msg = publish.call_args[0]
        self.assertEqual(topic, "/tf_static")
        self.assertIs(publish.call_args[1]["qos"], QoS.LATCHED)
        self.assertEqual(
            [t.child_frame_id for t in msg.transforms],
            ["test_entity1/odom", "other/odom"],
        )
        transform = msg.transforms[0]
        self.assertEqual(transform.header.frame_id, "world")
        self.assertEqual(transform.transform.translation.x, 1.0)
        self.assertEqual(transform.transform.translation.y, 2.0)
        self.assertEqual(transform.transform.translation.z, 3.0)
        self.assertEqual(transform.transform.rotation.w, 1.0)

    def test_call_with_retries_builds_native_request(self):
        response = MagicMock(success=True)
        self.mock_connector.sync.call.return_value = response
        result = self.bridge._call_with_retries(
            {"data": True}, target="/toggle", msg_type="std_srvs/srv/SetBool"
        )
        self.assertIs(result, response)
        target, request = self.mock_connector.sync.call.call_args[0]
        self.assertEqual(target, "/toggle")
        self.assertIsInstance(request, SetBool.Request)
        self.assertTrue(request.data)

    def test_clear_scene(self):
        self.bridge.spawned_entities = [
            self.test_spawned_entity,
            SpawnedEntity(
                id="entity_id_456",
                name="test_entity2",
                prefab_name="cube",
                pose=self.test_spawned_entity.pose,
            ),
        ]

        def side_effect(entity):
            if entity in self.bridge.spawned_entities:
                self.bridge.spawned_entities.remove(entity)

        self.bridge._despawn_entity = MagicMock(side_effect=side_effect)

        # Call clear_scene
        self.bridge.clear_scene()

        # Verify all entities were despawned
        self.assertEqual(len(self.bridge.spawned_entities), 0)
        # Verify _despawn_entity was called twice
        self.assertEqual(self.bridge._despawn_entity.call_count, 2)


def test_connector_has_the_methods_the_bridge_uses():
    from rai.communication.ros2.connectors.ros2_connector import SyncROS2Connector

    for name in ("now", "get_topics_names_and_types", "get_services_names_and_types"):
        assert callable(getattr(ROS2Connector, name)), name
    for name in ("call", "publish", "transform"):
        assert callable(getattr(SyncROS2Connector, name)), name
