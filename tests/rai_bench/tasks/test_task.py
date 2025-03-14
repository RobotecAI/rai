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

from typing import Any, Dict, List, Set

from rai_bench.benchmark_model import Task  # type: ignore
from rai_sim.simulation_bridge import Entity, Pose, Translation  # type: ignore
from tests.rai_bench.conftest import create_entity


class DummyTask(Task):
    def get_prompt(self) -> str:
        return "dummy prompt"

    def validate_config(self, simulation_config: Any) -> bool:
        return True

    def calculate_result(self, simulation_bridge: Any) -> float:
        return 1.0


def create_pose(x: float, y: float, z: float) -> Pose:
    return Pose(translation=Translation(x=x, y=y, z=z))


def test_build_neighbourhood_list() -> None:
    task = DummyTask()
    e1: Entity = create_entity("e1", "red_cube", 0, 0, 0)
    e2: Entity = create_entity("e2", "red_cube", 0.1, 0, 0)
    e3: Entity = create_entity("e3", "red_cube", 1, 1, 1)
    entities: List[Entity] = [e1, e2, e3]

    neighbourhood: Dict[Entity, List[Entity]] = task.build_neighbourhood_list(
        entities, threshold_distance=0.2
    )
    # e1 and e2 should be neighbours of each other; e3 remains isolated.
    assert set(neighbourhood[e1]) == {e2}, neighbourhood[e1]
    assert set(neighbourhood[e2]) == {e1}
    assert neighbourhood[e3] == []


def test_check_neighbourhood_types() -> None:
    task = DummyTask()

    e1: Entity = create_entity("e1", "red_cube", 0, 0, 0)
    e2: Entity = create_entity("e2", "red_cube", 0, 0, 0)

    assert task.check_neighbourhood_types([e1, e2], allowed_types=["red_cube"]) is True
    assert (
        task.check_neighbourhood_types([e1, e2], allowed_types=["blue_cube"]) is False
    )
    assert task.check_neighbourhood_types([], allowed_types=["red_cube"]) is True


def test_find_clusters() -> None:
    task = DummyTask()
    e1: Entity = create_entity("e1", "red_cube", 0, 0, 0)
    e2: Entity = create_entity("e2", "red_cube", 0, 0, 0)
    e3: Entity = create_entity("e3", "red_cube", 0, 0, 0)
    e4: Entity = create_entity("e4", "red_cube", 0, 0, 0)
    # Manually create a neighbourhood graph:
    neighbourhood: Dict[Entity, List[Entity]] = {
        e1: [e2],
        e2: [e1, e3],
        e3: [e2],
        e4: [],
    }
    clusters: List[List[Entity]] = task.find_clusters(neighbourhood)
    # Convert  to sets for order-independent comparison.
    clusters_as_sets: List[Set[Entity]] = [set(cluster) for cluster in clusters]
    assert {e1, e2, e3} in clusters_as_sets
    assert {e4} in clusters_as_sets
    assert len(clusters_as_sets) == 2


def test_group_entities_by_z_coordinate_all_stacked() -> None:
    task = DummyTask()
    e1: Entity = create_entity("e1", "red_cube", 0, 0, 0.0)
    e2: Entity = create_entity("e2", "red_cube", 0, 0, 0.05)
    e3: Entity = create_entity("e3", "red_cube", 0, 0, 0.2)
    e4: Entity = create_entity("e4", "red_cube", 0, 0, 0.25)
    e5: Entity = create_entity("e5", "red_cube", 0, 0, 0.5)
    entities: List[Entity] = [e1, e2, e3, e4, e5]

    groups: List[List[Entity]] = task.group_entities_along_z_axis(entities, margin=0.1)
    assert len(groups) == 1
    assert groups[0] == [e1, e2, e3, e4, e5]


def test_group_entities_by_z_coordinate_2_stacks() -> None:
    task = DummyTask()
    e1: Entity = create_entity("e1", "red_cube", 0, 1, 0.0)
    e2: Entity = create_entity("e2", "red_cube", 0, 1, 0.05)

    e3: Entity = create_entity("e3", "red_cube", 0, 0, 0.0)
    e4: Entity = create_entity("e4", "red_cube", 0, 0, 0.05)
    e5: Entity = create_entity("e5", "red_cube", 0, 0, 0.1)
    entities: List[Entity] = [e1, e2, e3, e4, e5]

    groups: List[List[Entity]] = task.group_entities_along_z_axis(entities, margin=0.01)
    # Convert to sets for order-independent comparison.
    groups_as_sets: List[Set[Entity]] = [set(group) for group in groups]
    assert len(groups) == 2
    assert {e1, e2} in groups_as_sets
    assert {e3, e4, e5} in groups_as_sets
