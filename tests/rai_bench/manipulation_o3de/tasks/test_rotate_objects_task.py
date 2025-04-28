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

import math

from rai.types import Quaternion

from rai_bench.manipulation_o3de.tasks import RotateObjectTask
from tests.rai_bench.conftest import create_entity


def test_calculate_perfect_match() -> None:
    target = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    task = RotateObjectTask(["apple"], target_quaternion=target)
    e1 = create_entity(
        "obj1", "apple", 0.3, 0.0, 0.05, Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    )
    correct, incorrect = task.calculate_correct([e1], allowable_rotation_error=5.0)
    assert correct == 1
    assert incorrect == 0


def test_calculate_error_under_threshold() -> None:
    target = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    task = RotateObjectTask(["apple"], target_quaternion=target)
    half_angle = math.radians(3.0)
    current_rotation = Quaternion(
        x=0.0, y=0.0, z=math.sin(half_angle), w=math.cos(half_angle)
    )
    e1 = create_entity("obj1", "apple", 0.3, 0.0, 0.05, current_rotation)
    # rotation error less then margin
    correct, incorrect = task.calculate_correct([e1], allowable_rotation_error=7.0)
    assert correct == 1
    assert incorrect == 0


def test_calculate_multiple_types() -> None:
    target = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    task = RotateObjectTask(["apple", "carrot"], target_quaternion=target)
    half_angle = math.radians(1.0)
    current_rotation = Quaternion(
        x=0.0, y=0.0, z=math.sin(half_angle), w=math.cos(half_angle)
    )
    e1 = create_entity("obj1", "apple", 0.3, 0.0, 0.05, current_rotation)
    e2 = create_entity("obj2", "apple", 0.4, 0.1, 0.05, current_rotation)
    e3 = create_entity("obj3", "carrot", 0.3, 0.0, 0.05, current_rotation)
    correct, incorrect = task.calculate_correct(
        [e1, e2, e3], allowable_rotation_error=5.0
    )
    assert correct == 3
    assert incorrect == 0


def test_calculate_mixed_types() -> None:
    target = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    task = RotateObjectTask(["yellow_cube", "carrot"], target_quaternion=target)
    half_angle = math.radians(1.0)
    current_rotation = Quaternion(
        x=0.0, y=0.0, z=math.sin(half_angle), w=math.cos(half_angle)
    )
    e1 = create_entity("obj1", "apple", 0.3, 0.0, 0.05, current_rotation)
    e2 = create_entity("obj2", "yellow_cube", 0.4, 0.1, 0.05, current_rotation)
    e3 = create_entity("obj3", "carrot", 0.3, 0.0, 0.05, current_rotation)
    correct, incorrect = task.calculate_correct(
        [e1, e2, e3], allowable_rotation_error=5.0
    )
    assert correct == 2
    assert incorrect == 0


def test_calculate_error_above_threshold() -> None:
    target = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    task = RotateObjectTask(["apple"], target_quaternion=target)
    half_angle = math.radians(5)
    current_rotation = Quaternion(
        x=0.0, y=0.0, z=math.sin(half_angle), w=math.cos(half_angle)
    )
    e1 = create_entity("obj1", "apple", 0.3, 0.0, 0.05, current_rotation)
    correct, incorrect = task.calculate_correct([e1], allowable_rotation_error=5.0)
    # The rotation error is 10°, so it exceeds the 5° threshold.
    assert correct == 0
    assert incorrect == 1
