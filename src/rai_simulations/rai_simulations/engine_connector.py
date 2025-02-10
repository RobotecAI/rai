# Copyright (C) 2024 Robotec.AI
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

from abc import ABC, abstractmethod
from geometry_msgs.msg import Pose

class Entity:
    # name: str
    # prefab_name: str
    # pose: Pose
    pass


class SceneConfig(BaseModel):
    """
    Setup of scene - arrangmenet of objects, interactions, environment etc.
    """

    entities: list[Entity]


class SceneSetup(ABC):
    """
    Info about entities in the scene (positions, collisions, etc.)
    """

    entities: list[Entity]


class EngineConnector(ABC):
    """
    Responsible for communication with simulation.
    """

    def __init__(self):
        pass

    @abstractmethod
    def setup_scene(self, scene_config: SceneConfig) -> SceneSetup:
        pass

    @abstractmethod
    def _spawn_entity(self, entity: Entity):
        pass

    @abstractmethod
    def despawn_entity(self, entity: Entity):
        pass

    @abstractmethod
    def get_object_position(self, object_name: str) -> Pose:
        pass


class O3DEEngineConnector(EngineConnector):
    def _spawn_entity(self, entity: Entity):
        # connector.service_call('spawn', entity)
        pass

    def _despawn_entity(self, entity: Entity):
        pass

    def setup_scene(self, scene_config: SceneConfig) -> SceneSetup:
        pass
        # 8 times despawn_entity
        # 10 times spawn_entity