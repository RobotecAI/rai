from pathlib import Path
from typing import Optional

import pytest
from pydantic import ValidationError

from rai_sim.simulation_bridge import (
    Entity,
    PoseModel,
    Rotation,
    SimulationConfig,
    SpawnedEntity,
    Translation,
)


# Helper Functions
def create_translation(x: float, y: float, z: float) -> Translation:
    return Translation(x=x, y=y, z=z)


def create_rotation(x: float, y: float, z: float, w: float) -> Rotation:
    return Rotation(x=x, y=y, z=z, w=w)


def create_pose(
    translation: Translation, rotation: Optional[Rotation] = None
) -> PoseModel:
    return PoseModel(translation=translation, rotation=rotation)


# Test Cases
def test_translation():
    translation = create_translation(x=1.1, y=2.2, z=3.3)

    assert isinstance(translation.x, float)
    assert isinstance(translation.y, float)
    assert isinstance(translation.z, float)

    assert translation.x == 1.1
    assert translation.y == 2.2
    assert translation.z == 3.3


def test_rotation():
    rotation = Rotation(x=0.1, y=0.2, z=0.3, w=0.4)

    assert isinstance(rotation.x, float)
    assert isinstance(rotation.y, float)
    assert isinstance(rotation.z, float)
    assert isinstance(rotation.w, float)

    assert rotation.x == 0.1
    assert rotation.y == 0.2
    assert rotation.z == 0.3
    assert rotation.w == 0.4


def test_pose():
    translation = create_translation(x=1.1, y=2.2, z=3.3)
    rotation = Rotation(x=0.1, y=0.2, z=0.3, w=0.4)

    pose = PoseModel(translation=translation, rotation=rotation)

    assert isinstance(pose.translation, Translation)
    assert isinstance(pose.rotation, Rotation)

    assert pose.translation.x == 1.1
    assert pose.rotation.w == 0.4


def test_optional_rotation():
    translation = create_translation(x=1.1, y=2.2, z=3.3)
    pose = create_pose(translation=translation)

    assert isinstance(pose.translation, Translation)
    assert pose.rotation is None


def test_entity():
    translation = create_translation(x=1.1, y=2.2, z=3.3)
    rotation = create_rotation(x=0.1, y=0.2, z=0.3, w=0.4)
    pose = create_pose(translation=translation, rotation=rotation)

    entity = Entity(name="test_cube", prefab_name="cube", pose=pose)

    assert isinstance(entity.name, str)
    assert isinstance(entity.prefab_name, str)
    assert isinstance(entity.pose, PoseModel)

    assert entity.name == "test_cube"
    assert entity.prefab_name == "cube"
    assert entity.pose == pose


def test_spawned_entity():
    translation = create_translation(x=1.1, y=2.2, z=3.3)
    rotation = create_rotation(x=0.1, y=0.2, z=0.3, w=0.4)
    pose = create_pose(translation=translation, rotation=rotation)

    spawned_entity = SpawnedEntity(
        name="test_cube",
        prefab_name="cube",
        pose=pose,
        id="id_123",
    )

    assert isinstance(spawned_entity.name, str)
    assert isinstance(spawned_entity.prefab_name, str)
    assert isinstance(spawned_entity.pose, PoseModel)
    assert isinstance(spawned_entity.id, str)

    assert spawned_entity.id == "id_123"


def test_simulation_config_unique_names():
    translation = create_translation(x=1.1, y=2.2, z=3.3)
    rotation = create_rotation(x=0.1, y=0.2, z=0.3, w=0.4)
    pose = create_pose(translation=translation, rotation=rotation)

    entities = [
        Entity(name="entity1", prefab_name="cube", pose=pose),
        Entity(name="entity2", prefab_name="carrot", pose=pose),
    ]

    config = SimulationConfig(entities=entities)

    assert isinstance(config.entities, list)
    assert all(isinstance(e, Entity) for e in config.entities)

    assert len(config.entities) == 2


def test_simulation_config_duplicate_names():
    translation = create_translation(x=1.1, y=2.2, z=3.3)
    rotation = create_rotation(x=0.1, y=0.2, z=0.3, w=0.4)
    pose = create_pose(translation=translation, rotation=rotation)

    entities = [
        Entity(name="duplicate", prefab_name="cube", pose=pose),
        Entity(name="duplicate", prefab_name="carrot", pose=pose),
    ]

    with pytest.raises(ValidationError):
        SimulationConfig(entities=entities)


# Test Reading from YAML File
@pytest.fixture
def sample_yaml_config(tmp_path: Path) -> Path:
    yaml_content = """
    entities:
      - name: entity1
        prefab_name: cube
        pose:
          translation:
            x: 1.0
            y: 2.0
            z: 3.0

      - name: entity2
        prefab_name: carrot
        pose:
          translation:
            x: 1.0
            y: 2.0
            z: 3.0
          rotation:
            x: 0.1
            y: 0.2
            z: 0.3
            w: 0.4
    """
    file_path = tmp_path / "test_config.yaml"
    file_path.write_text(yaml_content)
    return file_path


def test_load_base_config(sample_yaml_config: Path):
    config = SimulationConfig.load_base_config(sample_yaml_config)

    assert isinstance(config.entities, list)
    assert all(isinstance(e, Entity) for e in config.entities)

    assert len(config.entities) == 2
