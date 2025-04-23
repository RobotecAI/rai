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

from rai_bench.manipulation_o3de.tasks import PlaceCubesTask
from tests.rai_bench.conftest import create_entity


def test_calculate_all_adjacent() -> None:
    task = PlaceCubesTask(threshold_distance=0.15)
    # Create three cubes that are all close to each other.
    e1 = create_entity("cube1", "red_cube", 0.0, 0.0, 0.0)
    e2 = create_entity("cube2", "red_cube", 0.1, 0.1, 0.0)
    e3 = create_entity("cube3", "red_cube", 0.2, 0.2, 0.0)
    correct, incorrect = task.calculate_correct([e1, e2, e3])
    assert correct == 3
    assert incorrect == 0


def test_calculate_one_separated() -> None:
    task = PlaceCubesTask(threshold_distance=0.15)
    # Two cubes close together and one isolated.
    e1 = create_entity("cube1", "red_cube", 0.0, 0.0, 0.0)
    e2 = create_entity("cube2", "red_cube", 0.1, 0.1, 0.0)
    e3 = create_entity("cube3", "red_cube", 1.0, 1.0, 0.0)  # Isolated cube.
    correct, incorrect = task.calculate_correct([e1, e2, e3])
    assert correct == 2
    assert incorrect == 1


def test_calculate_none_adjacent() -> None:
    task = PlaceCubesTask(threshold_distance=0.15)
    # All cubes are far apart.
    e1 = create_entity("cube1", "red_cube", 0.0, 0.0, 0.0)
    e2 = create_entity("cube2", "red_cube", 1.0, 1.0, 0.0)
    e3 = create_entity("cube3", "red_cube", 2.0, 2.0, 0.0)
    correct, incorrect = task.calculate_correct([e1, e2, e3])
    assert correct == 0
    assert incorrect == 3


def test_calculate_2_clusters_adjacent() -> None:
    task = PlaceCubesTask(threshold_distance=0.15)
    # All cubes form 2 clusters
    e1 = create_entity("cube1", "red_cube", 0.0, 0.0, 0.0)
    e2 = create_entity("cube2", "red_cube", 0.1, 0.0, 0.0)
    e3 = create_entity("cube3", "red_cube", 2.0, 2.0, 0.0)
    e4 = create_entity("cube4", "red_cube", 2.1, 2.0, 0.0)
    correct, incorrect = task.calculate_correct([e1, e2, e3, e4])
    assert correct == 4
    assert incorrect == 0
