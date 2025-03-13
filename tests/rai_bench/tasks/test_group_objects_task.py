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

from typing import List

from rai_bench.o3de_test_bench.tasks import GroupObjectsTask  # type: ignore
from rai_sim.simulation_bridge import Entity  # type: ignore
from tests.rai_bench.conftest import create_entity


def test_calculate_correct_proper_cluster() -> None:
    task = GroupObjectsTask(["red_cube"])
    # Create three red_cube objects, all placed close together.
    e1 = create_entity("e1", "red_cube", 0.0, 0.0, 0.0)
    e2 = create_entity("e2", "red_cube", 0.1, 0.0, 0.0)
    e3 = create_entity("e3", "red_cube", 0.05, 0.05, 0.0)
    entities: List[Entity] = [e1, e2, e3]

    correct, misclustered = task.calculate_correct(entities)
    # Since all three red_cube objects form a single cluster, they are all considered correctly clustered.
    assert correct == 3
    assert misclustered == 0


def test_calculate_correct_multiple_clusters() -> None:
    task = GroupObjectsTask(["red_cube"])
    # Cluster 1
    e1 = create_entity("e1", "red_cube", 0.0, 0.0, 0.0)
    e2 = create_entity("e2", "red_cube", 0.05, 0.0, 0.0)
    # Cluster 2
    e3 = create_entity("e3", "red_cube", 5.0, 5.0, 0.0)
    e4 = create_entity("e4", "red_cube", 5.05, 5.05, 0.0)
    entities: List[Entity] = [e1, e2, e3, e4]

    correct, misclustered = task.calculate_correct(entities)
    # Since there are two clusters for red_cube, all objects are considered misclustered.
    assert correct == 0
    assert misclustered == 4


def test_calculate_correct_multiple_types() -> None:
    task = GroupObjectsTask(["red_cube", "blue_cube"])
    # Red cubes: form a proper cluster.
    r1 = create_entity("r1", "red_cube", 0.0, 0.0, 0.0)
    r2 = create_entity("r2", "red_cube", 0.1, 0.0, 0.0)
    r3 = create_entity("r3", "red_cube", 0.05, 0.05, 0.0)
    # Blue cubes: placed far apart so that each becomes its own cluster.
    b1 = create_entity("b1", "blue_cube", 10.0, 10.0, 0.0)
    b2 = create_entity("b2", "blue_cube", 10.1, 10.1, 0.0)
    entities: List[Entity] = [r1, r2, r3, b1, b2]

    correct, misclustered = task.calculate_correct(entities)

    assert correct == 5
    assert misclustered == 0


def test_calculate_correct_multiple_types_mixed() -> None:
    task = GroupObjectsTask(["red_cube", "blue_cube"])
    # Red cubes: form a proper cluster.
    r1 = create_entity("r1", "red_cube", 0.0, 0.0, 0.0)
    r2 = create_entity("r2", "red_cube", 0.1, 0.0, 0.0)
    r3 = create_entity("r3", "red_cube", 0.05, 0.05, 0.0)
    # Blue cubes: placed near to red cluster, so they are not separate
    b1 = create_entity("b1", "blue_cube", 0.1, 0.05, 0.0)
    b2 = create_entity("b2", "blue_cube", 0.05, 0.0, 0.0)
    entities: List[Entity] = [r1, r2, r3, b1, b2]

    correct, misclustered = task.calculate_correct(entities)

    assert correct == 0
    assert misclustered == 5


def test_calculate_correct_other_types_mixed() -> None:
    task = GroupObjectsTask(["red_cube"])
    # Red cubes: form a proper cluster.
    r1 = create_entity("r1", "red_cube", 0.0, 0.0, 0.0)
    r2 = create_entity("r2", "red_cube", 0.1, 0.0, 0.0)
    r3 = create_entity("r3", "red_cube", 0.05, 0.05, 0.0)
    # Blue cubes: arent clustered but placed near to red cluster,
    # so the red cubes cluster does not contain only red cubes
    b1 = create_entity("b1", "blue_cube", 0.1, 0.05, 0.0)
    b2 = create_entity("b2", "blue_cube", 0.05, 0.0, 0.0)
    entities: List[Entity] = [r1, r2, r3, b1, b2]

    correct, misclustered = task.calculate_correct(entities)

    assert correct == 0
    assert misclustered == 3


def test_calculate_correct_no_selected_objects() -> None:
    task = GroupObjectsTask(["apple"])
    # Red cubes: form a proper cluster.
    r1 = create_entity("r1", "red_cube", 0.0, 0.0, 0.0)
    r2 = create_entity("r2", "red_cube", 0.1, 0.0, 0.0)
    r3 = create_entity("r3", "red_cube", 0.05, 0.05, 0.0)
    # Blue cubes: arent clustered but placed near to red cluster,
    # so the red cubes cluster does not contain only red cubes
    b1 = create_entity("b1", "blue_cube", 0.1, 0.05, 0.0)
    b2 = create_entity("b2", "blue_cube", 0.05, 0.0, 0.0)
    entities: List[Entity] = [r1, r2, r3, b1, b2]

    correct, misclustered = task.calculate_correct(entities)

    assert correct == 0
    assert misclustered == 0
