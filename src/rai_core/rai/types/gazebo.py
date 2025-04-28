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
from pydantic import Field

from .base import RaiBaseModel
from .geometry import Pose, PoseStamped


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


class SpawnEntityService(RaiBaseModel):
    name: str
    robot_namespace: str
    reference_frame: str
    initial_pose: Pose
    xml: str = Field(default="")
