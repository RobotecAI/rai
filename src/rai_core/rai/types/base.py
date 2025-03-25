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

from typing import Optional

from pydantic import BaseModel, Field


class Point(BaseModel):
    """
    Represents the position of an object in 3D space using
    x, y, and z coordinates.
    """

    x: float = Field(description="X coordinate in meters")
    y: float = Field(description="Y coordinate in meters")
    z: float = Field(description="Z coordinate in meters")


class Quaternion(BaseModel):
    """
    Represents a 3D rotation using quaternion representation.
    """

    x: float = Field(0.0, description="X component of the quaternion")
    y: float = Field(0.0, description="Y component of the quaternion")
    z: float = Field(0.0, description="Z component of the quaternion")
    w: float = Field(1.0, description="W component of the quaternion")


class Pose(BaseModel):
    """
    Represents the complete pose (position and orientation) of an object.
    """

    position: Point = Field(description="The position of the object in 3D space")
    orientation: Optional[Quaternion] = Field(
        default=None,
        description="The orientation of the object as a quaternion. Optional if orientation is not needed and default orientation is handled by the bridge",
    )


class Header(BaseModel):
    """
    Header for a ROS message
    """

    frame_id: str = Field(description="Reference frame of the message")


class PoseStamped(BaseModel):
    """
    Pose with a reference frame
    """

    header: Header = Field(description="Reference frame of the message")
    pose: Pose = Field(description="Pose of the entity")


class Entity(BaseModel):
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


class SpawnEntityService(BaseModel):
    name: str
    robot_namespace: str
    reference_frame: str
    initial_pose: Pose
    xml: str = Field(default="")
