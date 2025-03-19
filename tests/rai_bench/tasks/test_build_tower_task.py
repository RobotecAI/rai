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

from rai_bench.o3de_test_bench.tasks import BuildCubeTowerTask
from tests.rai_bench.conftest import create_entity


def test_calculate_proper_tower() -> None:
    task = BuildCubeTowerTask(["red_cube"])
    e1 = create_entity("cube1", "red_cube", 0.0, 0.0, 0.0)
    e2 = create_entity("cube2", "red_cube", 0.01, 0.01, 0.03)
    e3 = create_entity("cube3", "red_cube", 0.0, 0.0, 0.04)
    correct, incorrect = task.calculate_correct([e1, e2, e3])
    assert correct == 3
    assert incorrect == 0


def test_calculate_multiple_groups() -> None:
    task = BuildCubeTowerTask(["red_cube"])
    e1 = create_entity("cube1", "red_cube", 0.0, 0.0, 0.0)  # Group 1
    e2 = create_entity("cube2", "red_cube", 0.0, 0.0, 0.03)  # Group 1
    e3 = create_entity("cube3", "red_cube", 0.0, 0.0, 0.06)  # Group 1

    e4 = create_entity("cube4", "red_cube", 0.0, 1.0, 0.03)  # Group 2
    e5 = create_entity("cube5", "red_cube", 0.0, 1.0, 0.06)  # Group 2

    # correct objects should be 3 as highest tower is 3 cubes high
    correct, incorrect = task.calculate_correct([e1, e2, e3, e4, e5])
    assert correct == 3
    assert incorrect == 2


def test_calculate_invalid_entity() -> None:
    task = BuildCubeTowerTask(["red_cube"])
    e1 = create_entity("cube1", "red_cube", 0.0, 0.0, 0.0)
    e2 = create_entity("cube2", "yellow_cube", 0.0, 0.0, 0.03)
    e3 = create_entity("cube3", "red_cube", 0.0, 0.0, 0.06)
    correct, incorrect = task.calculate_correct([e1, e2, e3])
    # The presence of an invalid object causes all cubes to be marked as incorrect.
    assert correct == 0
    assert incorrect == 2


def test_calculate_single_entity() -> None:
    task = BuildCubeTowerTask(["red_cube"])
    e1 = create_entity("cube1", "red_cube", 0.0, 0.0, 0.0)
    correct, incorrect = task.calculate_correct([e1])
    # A single cube in a group is considered incorrectly placed.
    assert correct == 0
    assert incorrect == 1
