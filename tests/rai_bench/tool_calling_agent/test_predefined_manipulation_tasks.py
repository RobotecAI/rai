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

from typing import Any, Dict, List

import pytest
from rai.tools.ros2 import MoveToPointToolInput

from rai_bench.tool_calling_agent.interfaces import TaskArgs
from rai_bench.tool_calling_agent.predefined.manipulation_tasks import (
    BANANA_OBJECT,
    BANANA_POSITION,
    CUBE_OBJECT,
    CUBE_POSITION,
    FRONT_DISTANCE,
    LEFT_DISTANCE,
    MOVE_TO_DROP_COORDS,
    MOVE_TO_GRAB_COORDS,
    get_both_object_positions_ord_val,
    grab_banana_ord_val,
    grab_cube_ord_val,
    move_banana_front_ord_val,
    move_banana_left_ord_val,
    move_cube_front_ord_val,
    move_cube_left_ord_val,
    move_to_point_ord_val_drop,
    move_to_point_ord_val_grab,
)
from rai_bench.tool_calling_agent.tasks.manipulation import (
    GetObjectPositionsTask,
    GrabExistingObjectTask,
    MoveExistingObjectFrontTask,
    MoveExistingObjectLeftTask,
    MoveToPointTask,
)


@pytest.fixture
def objects() -> Dict[str, Any]:
    """Create test objects for manipulation tasks."""
    return {
        BANANA_OBJECT: [BANANA_POSITION],
        CUBE_OBJECT: [CUBE_POSITION],
    }


class TestMoveToPointTask:
    """Test MoveToPointTask validation."""

    def test_move_to_point_grab_valid(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "move_to_point",
                "args": MOVE_TO_GRAB_COORDS,
            }
        ]

        task = MoveToPointTask(
            objects=objects,
            move_to_tool_input=MoveToPointToolInput(x=1.0, y=2.0, z=3.0, task="grab"),
            validators=[move_to_point_ord_val_grab],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_move_to_point_drop_valid(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "move_to_point",
                "args": MOVE_TO_DROP_COORDS,
            }
        ]

        task = MoveToPointTask(
            objects=objects,
            move_to_tool_input=MoveToPointToolInput(x=1.2, y=2.3, z=3.4, task="drop"),
            validators=[move_to_point_ord_val_drop],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_move_to_point_wrong_coordinates(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "move_to_point",
                "args": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "task": "grab",
                },  # Wrong coordinates
            }
        ]

        task = MoveToPointTask(
            objects=objects,
            move_to_tool_input=MoveToPointToolInput(x=1.0, y=2.0, z=3.0, task="grab"),
            validators=[move_to_point_ord_val_grab],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_move_to_point_wrong_task(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "move_to_point",
                "args": {"x": 1.0, "y": 2.0, "z": 3.0, "task": "drop"},  # Wrong task
            }
        ]

        task = MoveToPointTask(
            objects=objects,
            move_to_tool_input=MoveToPointToolInput(x=1.0, y=2.0, z=3.0, task="grab"),
            validators=[move_to_point_ord_val_grab],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_move_to_point_wrong_tool(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "wrong_tool_name",
                "args": MOVE_TO_GRAB_COORDS,
            }
        ]

        task = MoveToPointTask(
            objects=objects,
            move_to_tool_input=MoveToPointToolInput(x=1.0, y=2.0, z=3.0, task="grab"),
            validators=[move_to_point_ord_val_grab],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestGetObjectPositionsTask:
    """Test GetObjectPositionsTask validation."""

    def test_get_object_positions_valid_with_object_name(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_object_positions", "args": {"object_name": BANANA_OBJECT}},
            {"name": "get_object_positions", "args": {"object_name": CUBE_OBJECT}},
        ]

        task = GetObjectPositionsTask(
            objects=objects,
            validators=[get_both_object_positions_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_get_object_positions_missing_object(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_object_positions", "args": {"object_name": BANANA_OBJECT}},
            {"name": "get_object_positions", "args": {"object_name": BANANA_OBJECT}},
        ]

        task = GetObjectPositionsTask(
            objects=objects,
            validators=[get_both_object_positions_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_get_object_positions_wrong_tool(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [{"name": "wrong_tool_name", "args": {}}]

        task = GetObjectPositionsTask(
            objects=objects,
            validators=[get_both_object_positions_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_get_object_positions_unexpected_args(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_object_positions", "args": {"unexpected": "arg"}}
        ]

        task = GetObjectPositionsTask(
            objects=objects,
            validators=[get_both_object_positions_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestGrabExistingObjectTask:
    """Test GrabExistingObjectTask validation."""

    def test_grab_cube_valid(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_object_positions", "args": {"object_name": CUBE_OBJECT}},
            {
                "name": "move_to_point",
                "args": {
                    "x": CUBE_POSITION.x,
                    "y": CUBE_POSITION.y,
                    "z": CUBE_POSITION.z,
                    "task": "grab",
                },
            },
        ]

        task = GrabExistingObjectTask(
            objects=objects,
            object_to_grab=CUBE_OBJECT,
            validators=[grab_cube_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_grab_banana_valid(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_object_positions", "args": {"object_name": BANANA_OBJECT}},
            {
                "name": "move_to_point",
                "args": {
                    "x": BANANA_POSITION.x,
                    "y": BANANA_POSITION.y,
                    "z": BANANA_POSITION.z,
                    "task": "grab",
                },
            },
        ]

        task = GrabExistingObjectTask(
            objects=objects,
            object_to_grab=BANANA_OBJECT,
            validators=[grab_banana_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_grab_wrong_coordinates(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_object_positions", "args": {"object_name": CUBE_OBJECT}},
            {
                "name": "move_to_point",
                "args": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "task": "grab",
                },  # Wrong coordinates
            },
        ]

        task = GrabExistingObjectTask(
            objects=objects,
            object_to_grab=CUBE_OBJECT,
            validators=[grab_cube_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_grab_missing_get_positions(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "move_to_point",
                "args": {
                    "x": CUBE_POSITION.x,
                    "y": CUBE_POSITION.y,
                    "z": CUBE_POSITION.z,
                    "task": "grab",
                },
            },
        ]

        task = GrabExistingObjectTask(
            objects=objects,
            object_to_grab=CUBE_OBJECT,
            validators=[grab_cube_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_grab_wrong_order(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "move_to_point",
                "args": {
                    "x": CUBE_POSITION.x,
                    "y": CUBE_POSITION.y,
                    "z": CUBE_POSITION.z,
                    "task": "grab",
                },
            },
            {
                "name": "get_object_positions",
                "args": {"object_name": CUBE_OBJECT},
            },  # Wrong order
        ]

        task = GrabExistingObjectTask(
            objects=objects,
            object_to_grab=CUBE_OBJECT,
            validators=[grab_cube_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestMoveExistingObjectLeftTask:
    """Test MoveExistingObjectLeftTask validation."""

    def test_move_cube_left_valid(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_object_positions", "args": {"object_name": CUBE_OBJECT}},
            {
                "name": "move_to_point",
                "args": {
                    "x": CUBE_POSITION.x,
                    "y": CUBE_POSITION.y,
                    "z": CUBE_POSITION.z,
                    "task": "grab",
                },
            },
            {
                "name": "move_to_point",
                "args": {
                    "x": CUBE_POSITION.x,
                    "y": round(CUBE_POSITION.y - LEFT_DISTANCE, 2),
                    "z": CUBE_POSITION.z,
                    "task": "drop",
                },
            },
        ]

        task = MoveExistingObjectLeftTask(
            objects=objects,
            object_to_grab=CUBE_OBJECT,
            validators=[move_cube_left_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_move_banana_left_valid(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_object_positions", "args": {"object_name": BANANA_OBJECT}},
            {
                "name": "move_to_point",
                "args": {
                    "x": BANANA_POSITION.x,
                    "y": BANANA_POSITION.y,
                    "z": BANANA_POSITION.z,
                    "task": "grab",
                },
            },
            {
                "name": "move_to_point",
                "args": {
                    "x": BANANA_POSITION.x,
                    "y": round(BANANA_POSITION.y - LEFT_DISTANCE, 2),
                    "z": BANANA_POSITION.z,
                    "task": "drop",
                },
            },
        ]

        task = MoveExistingObjectLeftTask(
            objects=objects,
            object_to_grab=BANANA_OBJECT,
            validators=[move_banana_left_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_move_left_wrong_target_position(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_object_positions", "args": {"object_name": CUBE_OBJECT}},
            {
                "name": "move_to_point",
                "args": {
                    "x": CUBE_POSITION.x,
                    "y": CUBE_POSITION.y,
                    "z": CUBE_POSITION.z,
                    "task": "grab",
                },
            },
            {
                "name": "move_to_point",
                "args": {
                    "x": CUBE_POSITION.x,
                    "y": CUBE_POSITION.y,
                    "z": CUBE_POSITION.z,
                    "task": "drop",
                },  # Same position, not left
            },
        ]

        task = MoveExistingObjectLeftTask(
            objects=objects,
            object_to_grab=CUBE_OBJECT,
            validators=[move_cube_left_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_move_left_missing_drop(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_object_positions", "args": {"object_name": CUBE_OBJECT}},
            {
                "name": "move_to_point",
                "args": {
                    "x": CUBE_POSITION.x,
                    "y": CUBE_POSITION.y,
                    "z": CUBE_POSITION.z,
                    "task": "grab",
                },  # missing drop
            },
        ]

        task = MoveExistingObjectLeftTask(
            objects=objects,
            object_to_grab=CUBE_OBJECT,
            validators=[move_cube_left_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestMoveExistingObjectFrontTask:
    """Test MoveExistingObjectFrontTask validation."""

    def test_move_cube_front_valid(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_object_positions", "args": {"object_name": CUBE_OBJECT}},
            {
                "name": "move_to_point",
                "args": {
                    "x": CUBE_POSITION.x,
                    "y": CUBE_POSITION.y,
                    "z": CUBE_POSITION.z,
                    "task": "grab",
                },
            },
            {
                "name": "move_to_point",
                "args": {
                    "x": round(CUBE_POSITION.x + FRONT_DISTANCE, 2),
                    "y": CUBE_POSITION.y,
                    "z": CUBE_POSITION.z,
                    "task": "drop",
                },
            },
        ]

        task = MoveExistingObjectFrontTask(
            objects=objects,
            object_to_grab=CUBE_OBJECT,
            validators=[move_cube_front_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_move_banana_front_valid(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_object_positions", "args": {"object_name": BANANA_OBJECT}},
            {
                "name": "move_to_point",
                "args": {
                    "x": BANANA_POSITION.x,
                    "y": BANANA_POSITION.y,
                    "z": BANANA_POSITION.z,
                    "task": "grab",
                },
            },
            {
                "name": "move_to_point",
                "args": {
                    "x": round(BANANA_POSITION.x + FRONT_DISTANCE, 2),
                    "y": BANANA_POSITION.y,
                    "z": BANANA_POSITION.z,
                    "task": "drop",
                },
            },
        ]

        task = MoveExistingObjectFrontTask(
            objects=objects,
            object_to_grab=BANANA_OBJECT,
            validators=[move_banana_front_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_move_front_wrong_direction(
        self, task_args: TaskArgs, objects: Dict[str, Any]
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_object_positions", "args": {"object_name": CUBE_OBJECT}},
            {
                "name": "move_to_point",
                "args": {
                    "x": CUBE_POSITION.x,
                    "y": CUBE_POSITION.y,
                    "z": CUBE_POSITION.z,
                    "task": "grab",
                },
            },
            {
                "name": "move_to_point",
                "args": {
                    "x": CUBE_POSITION.x - FRONT_DISTANCE,
                    "y": CUBE_POSITION.y,
                    "z": CUBE_POSITION.z,
                    "task": "drop",
                },  # Wrong direction (back instead of front)
            },
        ]

        task = MoveExistingObjectFrontTask(
            objects=objects,
            object_to_grab=CUBE_OBJECT,
            validators=[move_cube_front_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0
