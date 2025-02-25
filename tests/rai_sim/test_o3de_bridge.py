import signal
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from geometry_msgs.msg import Point, Pose, Quaternion
from rai.communication.ros2.connectors import ROS2ARIConnector

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
