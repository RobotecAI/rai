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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, List, Optional, TypeVar

import yaml
from pydantic import BaseModel, Field, field_validator
from rai.types import (
    Header,
    Point,
    Pose,
    PoseStamped,
    Quaternion,
    RaiBaseModel,
    ROS2BaseModel,
)


class Entity(RaiBaseModel):
    """
    Entity that can be spawned in the simulation environment.
    """

    name: str = Field(description="Unique name for the entity")
    prefab_name: str = Field(
        description="Name of the prefab resource to use for spawning this entity"
    )
    pose: PoseStamped = Field(description="Initial pose of the entity")

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if isinstance(other, Entity) or isinstance(other, SpawnedEntity):
            return self.name == other.name
        else:
            return False


class SpawnedEntity(Entity):
    """
    Entity that has been spawned in the simulation environment.
    """

    id: str = Field(
        description="Unique identifier assigned to the spawned entity instance"
    )


class SpawnEntityService(ROS2BaseModel):
    _prefix: str = "gazebo_msgs/srv"
    name: str
    robot_namespace: str
    reference_frame: str
    initial_pose: Pose
    xml: str = Field(default="")


class SceneConfig(BaseModel):
    """
    Setup of simulation - arrangement of objects in the environment.

    Attributes
    ----------
    List of entities to be spawned in the simulation.
    """

    entities: List[Entity] = Field(
        description="List of entities to be spawned in the simulation environment"
    )

    @field_validator("entities")
    @classmethod
    def check_unique_names(cls, entities: List[Entity]) -> List[Entity]:
        """
        Validates that all entity names in the configuration are unique.

        Parameters
        ----------
        entities : List[Entity]
            List of entities to validate.

        Returns
        -------
        List[Entity]
            The validated list of entities.

        Raises
        ------
        ValueError
            If any entity names are duplicated.
        """
        names = [entity.name for entity in entities]
        if len(names) != len(set(names)):
            raise ValueError("Each entity must have a unique name.")
        return entities

    @classmethod
    def load_base_config(cls, base_config_path: Path) -> "SceneConfig":
        """
        Loads a simulation configuration from a YAML file.

        Parameters
        ----------
        base_config_path : Path
            Path to the YAML configuration file.

        Returns
        -------
        SimulationConfig
            The loaded simulation configuration.
        """
        with open(base_config_path) as f:
            content = yaml.safe_load(f)
        frame_id = content.get("frame_id", "odom")
        entities = [
            Entity(
                name=entity["name"],
                prefab_name=entity["prefab_name"],
                pose=PoseStamped(
                    header=Header(frame_id=frame_id),
                    pose=Pose(
                        position=Point(
                            **entity["pose"]["translation"]
                        ),  # TODO(boczekbartek): adapt yaml configs to ros2 naming convension
                        orientation=Quaternion(
                            **entity["pose"].get("rotation", dict())
                        ),
                    ),
                ),
            )
            for entity in content["entities"]
        ]

        return cls(entities=entities)


class SceneState(BaseModel):
    """
    Info about current state of the scene.

    Attributes
    ----------
    entities : List[SpawnedEntity]
        List of all entities currently present in the scene.
    """

    entities: List[SpawnedEntity] = Field(
        description="List of all entities currently spawned in the scene with their current poses"
    )


class SimulationConfig(BaseModel): ...


SimulationConfigT = TypeVar("SimulationConfigT", bound=SimulationConfig)


class SimulationBridge(ABC, Generic[SimulationConfigT]):
    """
    Responsible for communication with simulation.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.spawned_entities: List[
            SpawnedEntity
        ] = []  # list of spawned entities with their initial poses
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    @abstractmethod
    def init_simulation(self, simulation_config: SimulationConfigT):
        """
        Initialize simulation binary and all other required processes,
        for example ros2 nodes
        """
        pass

    @abstractmethod
    def setup_scene(self, scene_config: SceneConfig):
        """
        Runs and sets up the simulation scene according to the provided configuration.

        Parameters
        ----------
        simulation_config : SimulationConfigT
            Configuration containing the simulation initialization and setup details including
            entities to be spawned and their initial poses.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def _spawn_entity(self, entity: Entity):
        """
        Spawns a single entity in the simulation environment.

        Parameters
        ----------
        entity : Entity
            Entity object containing the entity's properties

        Returns
        -------
        None

        Notes
        -----
        The spawned entity should be added to the spawned_entities list maintained
        by the simulation bridge.
        """
        pass

    @abstractmethod
    def _despawn_entity(self, entity: SpawnedEntity):
        """
        Removes a previously spawned entity from the simulation environment.

        Parameters
        ----------
        entity : SpawnedEntity
            Entity object representing the spawned entity to be removed.

        Returns
        -------
        None

        Notes
        -----
        Despawned entity should be removed from the spawned_entities list maintained
        by the simulation bridge.
        """
        pass

    @abstractmethod
    def get_object_pose(self, entity: SpawnedEntity) -> PoseStamped:
        """
        Gets the current pose of a spawned entity.

        This method queries the simulation to get the current position and
        orientation of a specific entity.

        Parameters
        ----------
        entity : SpawnedEntity
            Entity object representing the spawned entity whose pose is
            to be retrieved.

        Returns
        -------
        PoseStamped
            Object containing the entity's current pose.
        """
        pass

    @abstractmethod
    def get_scene_state(self) -> SceneState:
        """
        Gets the current state of the simulation scene.

        Parameters
        ----------
        None

        Returns
        -------
        SceneState
            Object containing the current state of the scene.

        Notes
        -----
        SceneState should contain the current poses of spawned_entities.
        """
        pass
