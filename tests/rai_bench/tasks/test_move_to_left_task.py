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

from rai_bench.o3de_test_bench.tasks import MoveObjectsToLeftTask
from tests.rai_bench.conftest import create_entity


def test_calculate_all_on_left() -> None:
    task = MoveObjectsToLeftTask(["red_cube", "blue_cube"])
    e1 = create_entity("obj1", "red_cube", 0.0, 1.0, 0.0)  # y > 0 → correct
    e2 = create_entity("obj2", "blue_cube", 0.0, 0.5, 0.0)  # y > 0 → correct
    correct, incorrect = task.calculate_correct([e1, e2])
    assert correct == 2
    assert incorrect == 0


def test_calculate_some_not_on_left() -> None:
    task = MoveObjectsToLeftTask(["red_cube", "blue_cube"])
    e1 = create_entity("obj1", "red_cube", 0.0, 1.0, 0.0)  # y > 0 → correct
    e2 = create_entity("obj2", "blue_cube", 0.0, -0.5, 0.0)  # y < 0 → incorrect
    correct, incorrect = task.calculate_correct([e1, e2])
    assert correct == 1
    assert incorrect == 1


def test_calculate_other_types() -> None:
    task = MoveObjectsToLeftTask(["red_cube", "blue_cube"])
    e1 = create_entity("obj1", "red_cube", 0.0, 1.0, 0.0)  # valid type, y > 0 → correct
    e2 = create_entity(
        "obj2", "apple", 0.0, 1.0, 0.0
    )  # invalid type, should be ignored
    correct, incorrect = task.calculate_correct([e1, e2])
    assert correct == 1
    assert incorrect == 0
