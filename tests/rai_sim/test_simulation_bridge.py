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
import unittest
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from rai.types import (
    Header,
    Point,
    Pose,
    PoseStamped,
    Quaternion,
)

from rai_sim.simulation_bridge import (
    Entity,
    SceneConfig,
    SceneState,
    SimulationBridge,
    SimulationConfig,
    SpawnedEntity,
)


# Test Cases
def test_position():
    position = Point(x=1.1, y=2.2, z=3.3)

    assert isinstance(position.x, float)
    assert isinstance(position.y, float)
    assert isinstance(position.z, float)

    assert np.isclose(position.x, 1.1)
    assert np.isclose(position.y, 2.2)
    assert np.isclose(position.z, 3.3)


def test_quaternion():
    quaternion = Quaternion(x=0.1, y=0.2, z=0.3, w=0.4)

    assert isinstance(quaternion.x, float)
    assert isinstance(quaternion.y, float)
    assert isinstance(quaternion.z, float)
    assert isinstance(quaternion.w, float)

    assert np.isclose(quaternion.x, 0.1)
    assert np.isclose(quaternion.y, 0.2)
    assert np.isclose(quaternion.z, 0.3)
    assert np.isclose(quaternion.w, 0.4)


def test_pose():
    position = Point(x=1.1, y=2.2, z=3.3)
    quaternion = Quaternion(x=0.1, y=0.2, z=0.3, w=0.4)

    pose = Pose(position=position, orientation=quaternion)

    assert isinstance(pose.position, Point)
    assert isinstance(pose.orientation, Quaternion)

    assert np.isclose(pose.position.x, 1.1)
    assert np.isclose(pose.orientation.w, 0.4)


@pytest.fixture
def pose() -> PoseStamped:
    position = Point(x=1.1, y=2.2, z=3.3)
    quaternion = Quaternion(x=0.1, y=0.2, z=0.3, w=0.4)
    header = Header(frame_id="test_frame")
    return PoseStamped(
        header=header, pose=Pose(position=position, orientation=quaternion)
    )


def test_entity(pose: PoseStamped):
    entity = Entity(name="test_cube", prefab_name="cube", pose=pose)

    assert isinstance(entity.name, str)
    assert isinstance(entity.prefab_name, str)
    assert isinstance(entity.pose, PoseStamped)

    assert entity.name == "test_cube"
    assert entity.prefab_name == "cube"
    assert entity.pose == pose


def test_spawned_entity(pose: PoseStamped):
    spawned_entity = SpawnedEntity(
        name="test_cube",
        prefab_name="cube",
        pose=pose,
        id="id_123",
    )

    assert isinstance(spawned_entity.name, str)
    assert isinstance(spawned_entity.prefab_name, str)
    assert isinstance(spawned_entity.pose, PoseStamped)
    assert isinstance(spawned_entity.id, str)

    assert spawned_entity.id == "id_123"


def test_scene_config_unique_names(pose: PoseStamped):
    entities = [
        Entity(name="entity1", prefab_name="cube", pose=pose),
        Entity(name="entity2", prefab_name="carrot", pose=pose),
    ]

    config = SceneConfig(entities=entities)

    assert isinstance(config.entities, list)
    assert all(isinstance(e, Entity) for e in config.entities)

    assert len(config.entities) == 2


def test_scene_config_duplicate_names(pose: PoseStamped):
    entities = [
        Entity(name="duplicate", prefab_name="cube", pose=pose),
        Entity(name="duplicate", prefab_name="carrot", pose=pose),
    ]

    with pytest.raises(ValidationError):
        SceneConfig(entities=entities)


def test_load_base_config(sample_base_yaml_config: Path):
    config = SceneConfig.load_base_config(sample_base_yaml_config)

    assert isinstance(config.entities, list)
    assert all(isinstance(e, Entity) for e in config.entities)
    assert all(e.pose._prefix == "geometry_msgs/msg" for e in config.entities)

    assert len(config.entities) == 2


class MockSimulationBridge(SimulationBridge):
    """Mock implementation of SimulationBridge for testing."""

    def init_simulation(self, simulation_config: SimulationConfig):
        pass

    def setup_scene(self, scene_config: SceneConfig):
        """Mock implementation of setup_scene."""
        for entity in scene_config.entities:
            self._spawn_entity(entity)

    def _spawn_entity(self, entity: Entity):
        """Mock implementation of _spawn_entity."""
        spawned_entity = SpawnedEntity(
            id=f"id_{entity.name}",
            name=entity.name,
            prefab_name=entity.prefab_name,
            pose=entity.pose,
        )
        self.spawned_entities.append(spawned_entity)

    def _despawn_entity(self, entity: SpawnedEntity):
        """Mock implementation of _despawn_entity."""
        self.spawned_entities = [e for e in self.spawned_entities if e.id != entity.id]

    def get_object_pose(self, entity: SpawnedEntity) -> PoseStamped:
        """Mock implementation of get_object_pose."""
        for spawned_entity in self.spawned_entities:
            if spawned_entity.id == entity.id:
                return spawned_entity.pose
        raise ValueError(f"Entity with id {entity.id} not found")

    def get_scene_state(self) -> SceneState:
        """Mock implementation of get_scene_state."""
        return SceneState(entities=self.spawned_entities)


class TestSimulationBridge(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test_logger")
        self.bridge = MockSimulationBridge(logger=self.logger)

        # Create test entities
        header = Header(frame_id="test_frame")
        self.test_entity1: Entity = Entity(
            name="test_entity1",
            prefab_name="test_prefab1",
            pose=PoseStamped(
                header=header,
                pose=Pose(
                    position=Point(x=1.0, y=2.0, z=3.0),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            ),
        )

        self.test_entity2: Entity = Entity(
            name="test_entity2",
            prefab_name="test_prefab2",
            pose=PoseStamped(
                header=header,
                pose=Pose(position=Point(x=4.0, y=5.0, z=6.0)),
            ),
        )

        # Create a test configuration
        self.test_config = SceneConfig(entities=[self.test_entity1, self.test_entity2])

    def test_init(self):
        # Test with provided logger
        bridge = MockSimulationBridge(logger=self.logger)
        self.assertEqual(bridge.logger, self.logger)
        self.assertEqual(len(bridge.spawned_entities), 0)

        # Test with default logger
        bridge = MockSimulationBridge()
        self.assertIsNotNone(bridge.logger)
        self.assertEqual(len(bridge.spawned_entities), 0)

    def test_setup_scene(self):
        self.bridge.setup_scene(self.test_config)

        # Check if entities were spawned
        self.assertEqual(len(self.bridge.spawned_entities), 2)

        # Check if the spawned entities have correct properties
        pose1 = self.bridge.spawned_entities[0].pose.pose
        self.assertEqual(self.bridge.spawned_entities[0].name, "test_entity1")
        self.assertEqual(self.bridge.spawned_entities[0].prefab_name, "test_prefab1")
        assert np.isclose(pose1.position.x, 1.0)
        assert np.isclose(pose1.position.y, 2.0)
        assert np.isclose(pose1.position.z, 3.0)
        assert pose1.orientation
        assert np.isclose(pose1.orientation.x, 0.0)
        assert np.isclose(pose1.orientation.y, 0.0)
        assert np.isclose(pose1.orientation.z, 0.0)
        assert np.isclose(pose1.orientation.w, 1.0)

        pose2 = self.bridge.spawned_entities[1].pose.pose
        self.assertEqual(self.bridge.spawned_entities[1].name, "test_entity2")
        self.assertEqual(self.bridge.spawned_entities[1].prefab_name, "test_prefab2")
        assert np.isclose(pose2.position.x, 4.0)
        assert np.isclose(pose2.position.y, 5.0)
        assert np.isclose(pose2.position.z, 6.0)
        assert np.isclose(pose2.orientation.x, 0.0)
        assert np.isclose(pose2.orientation.y, 0.0)
        assert np.isclose(pose2.orientation.z, 0.0)
        assert np.isclose(pose2.orientation.w, 1.0)

    def test_spawn_entity(self):
        self.bridge._spawn_entity(self.test_entity1)  # type: ignore
        spawned_entity = self.bridge.spawned_entities[0]
        # Check if entity was added to spawned_entities
        self.assertEqual(len(self.bridge.spawned_entities), 1)
        self.assertIsInstance(spawned_entity, SpawnedEntity)

    def test_despawn_entity(self):
        # First spawn an entity
        self.bridge._spawn_entity(self.test_entity1)  # type: ignore
        self.assertEqual(len(self.bridge.spawned_entities), 1)

        # Then despawn it
        self.bridge._despawn_entity(self.bridge.spawned_entities[0])  # type: ignore
        self.assertEqual(len(self.bridge.spawned_entities), 0)

    def test_get_object_pose(self):
        # First spawn an entity
        self.bridge._spawn_entity(self.test_entity1)  # type: ignore

        # Get the pose
        pose_stamped: PoseStamped = self.bridge.get_object_pose(
            self.bridge.spawned_entities[0]
        )

        # Check if the pose matches
        pose = pose_stamped.pose
        assert np.isclose(pose.position.x, 1.0)
        assert np.isclose(pose.position.y, 2.0)
        assert np.isclose(pose.position.z, 3.0)
        assert pose.orientation
        assert np.isclose(pose.orientation.x, 0.0)
        assert np.isclose(pose.orientation.y, 0.0)
        assert np.isclose(pose.orientation.z, 0.0)
        assert np.isclose(pose.orientation.w, 1.0)

        # Test for non-existent entity
        non_existent_entity = SpawnedEntity(
            id="non_existent",
            name="non_existent",
            prefab_name="non_existent",
            pose=PoseStamped(
                header=Header(frame_id="test_frame"),
                pose=Pose(position=Point(x=0.0, y=0.0, z=0.0)),
            ),
        )

        with self.assertRaises(ValueError):
            self.bridge.get_object_pose(non_existent_entity)

    def test_get_scene_state(self):
        # First spawn some entities
        self.bridge._spawn_entity(self.test_entity1)  # type: ignore
        self.bridge._spawn_entity(self.test_entity2)  # type: ignore

        # Get the scene state
        scene_state = self.bridge.get_scene_state()

        # Check if the scene state contains the correct entities
        self.assertEqual(len(scene_state.entities), 2)
        self.assertEqual(scene_state.entities[0].name, "test_entity1")
        self.assertEqual(scene_state.entities[1].name, "test_entity2")
