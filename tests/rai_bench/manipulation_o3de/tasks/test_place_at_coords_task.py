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

from rai_bench.manipulation_o3de_bench.tasks import PlaceObjectAtCoordTask
from tests.rai_bench.conftest import create_entity


def test_calculate_single_object_correct() -> None:
    task = PlaceObjectAtCoordTask("carrot", (0.5, 0.5))
    e1 = create_entity("carrot1", "carrot", 0.5, 0.5, 0.0)
    correct, incorrect = task.calculate_correct([e1])
    assert correct == 1
    assert incorrect == 0


def test_calculate_single_object_incorrect() -> None:
    task = PlaceObjectAtCoordTask("carrot", (0.5, 0.5))
    e1 = create_entity("carrot1", "carrot", 0.7, 0.7, 0.0)
    correct, incorrect = task.calculate_correct([e1])
    assert correct == 0
    assert incorrect == 1


def test_calculate_multiple_objects_one_correct() -> None:
    task = PlaceObjectAtCoordTask("carrot", (0.5, 0.5))
    e1 = create_entity("carrot1", "carrot", 0.5, 0.5, 0.0)  # Correct placement.
    e2 = create_entity("carrot2", "carrot", 0.6, 0.6, 0.0)  # Incorrect placement.
    correct, incorrect = task.calculate_correct([e1, e2])
    # only one is considered
    assert correct == 1
    assert incorrect == 0


def test_calculate_multiple_objects_multiple_correct() -> None:
    task = PlaceObjectAtCoordTask("carrot", (0.5, 0.5))
    e1 = create_entity("carrot1", "carrot", 0.5, 0.5, 0.0)  # Correct placement.
    e2 = create_entity(
        "carrot2", "carrot", 0.501, 0.501, 0.0
    )  # also correct placement.
    correct, incorrect = task.calculate_correct([e1, e2])
    assert correct == 1
    assert incorrect == 0


def test_calculate_multiple_objects_none_correct() -> None:
    task = PlaceObjectAtCoordTask("carrot", (0.5, 0.5))
    e1 = create_entity("carrot1", "carrot", 0.6, 0.6, 0.0)  # Off target.
    e2 = create_entity("carrot2", "carrot", 0.7, 0.7, 0.0)  # Off target.
    correct, incorrect = task.calculate_correct([e1, e2])
    assert correct == 0
    assert incorrect == 1
