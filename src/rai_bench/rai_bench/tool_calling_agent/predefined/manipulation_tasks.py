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

from typing import Any, Dict, List, Literal

from rai.tools.ros2 import MoveToPointToolInput
from rai.types import Point

from rai_bench.tool_calling_agent.interfaces import (
    Task,
    TaskArgs,
)
from rai_bench.tool_calling_agent.subtasks import (
    CheckArgsToolCallSubTask,
)
from rai_bench.tool_calling_agent.tasks.manipulation import (
    GetObjectPositionsTask,
    GrabExistingObjectTask,
    MoveExistingObjectFrontTask,
    MoveExistingObjectLeftTask,
    MoveToPointTask,
)
from rai_bench.tool_calling_agent.validators import (
    NotOrderedCallsValidator,
    OrderedCallsValidator,
)

BANANA_POSITION = Point(x=0.1, y=0.2, z=0.3)
BANANA_POSITION_2 = Point(x=0.4, y=0.5, z=0.6)
CUBE_POSITION = Point(x=0.7, y=0.8, z=0.9)

LEFT_DISTANCE = 0.2  # 20cm
FRONT_DISTANCE = 0.6  # 60cm

MOVE_TO_GRAB_COORDS: Dict[str, Any] = {"x": 1.0, "y": 2.0, "z": 3.0, "task": "grab"}
MOVE_TO_DROP_COORDS: Dict[str, Any] = {"x": 1.2, "y": 2.3, "z": 3.4, "task": "drop"}

BANANA_OBJECT = "banana"
CUBE_OBJECT = "cube"
APPLE_OBJECT = "apple"

########## SUBTASKS #################################################################
move_to_point_subtask_grab = CheckArgsToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args=MOVE_TO_GRAB_COORDS,
)
move_to_point_subtask_drop = CheckArgsToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args=MOVE_TO_DROP_COORDS,
)


get_object_positions_banana_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_object_positions",
    expected_args={"object_name": BANANA_OBJECT},
)

get_object_positions_cube_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_object_positions",
    expected_args={"object_name": CUBE_OBJECT},
)

grab_cube_move_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args={
        "x": CUBE_POSITION.x,
        "y": CUBE_POSITION.y,
        "z": CUBE_POSITION.z,
        "task": "grab",
    },
)
grab_banana_move_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args={
        "x": BANANA_POSITION.x,
        "y": BANANA_POSITION.y,
        "z": BANANA_POSITION.z,
        "task": "grab",
    },
)

move_cube_left_grab_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args={
        "x": CUBE_POSITION.x,
        "y": CUBE_POSITION.y,
        "z": CUBE_POSITION.z,
        "task": "grab",
    },
)
move_cube_left_drop_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args={
        "x": CUBE_POSITION.x,
        "y": round(CUBE_POSITION.y - LEFT_DISTANCE, 2),
        "z": CUBE_POSITION.z,
        "task": "drop",
    },
)

move_banana_left_grab_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args={
        "x": BANANA_POSITION.x,
        "y": BANANA_POSITION.y,
        "z": BANANA_POSITION.z,
        "task": "grab",
    },
)
move_banana_left_drop_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args={
        "x": BANANA_POSITION.x,
        "y": round(BANANA_POSITION.y - LEFT_DISTANCE, 2),
        "z": BANANA_POSITION.z,
        "task": "drop",
    },
)

move_cube_front_grab_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args={
        "x": CUBE_POSITION.x,
        "y": CUBE_POSITION.y,
        "z": CUBE_POSITION.z,
        "task": "grab",
    },
)
move_cube_front_drop_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args={
        "x": round(CUBE_POSITION.x + FRONT_DISTANCE, 2),
        "y": CUBE_POSITION.y,
        "z": CUBE_POSITION.z,
        "task": "drop",
    },
)

move_banana_front_grab_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args={
        "x": BANANA_POSITION.x,
        "y": BANANA_POSITION.y,
        "z": BANANA_POSITION.z,
        "task": "grab",
    },
)
move_banana_front_drop_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args={
        "x": round(BANANA_POSITION.x + FRONT_DISTANCE, 2),
        "y": BANANA_POSITION.y,
        "z": BANANA_POSITION.z,
        "task": "drop",
    },
)

# swap_objects_grab_cube_subtask = CheckArgsToolCallSubTask(
#     expected_tool_name="move_to_point",
#     expected_args={
#         "x": CUBE_POSITION.x,
#         "y": CUBE_POSITION.y,
#         "z": CUBE_POSITION.z,
#         "task": "grab",
#     },
# )
# swap_objects_move_cube_to_temp_subtask = CheckArgsToolCallSubTask(
#     expected_tool_name="move_to_point",
#     expected_args={
#         # NOTE (jmatejcz) technically we should not accept any position but
#         # every besides the positions of the objects to swap
#         # but we currently don't implement checks for negative validation
#         # so just accept any coords
#         "task": "drop",
#     },
#     expected_optional_args={
#         "x": float,
#         "y": float,
#         "z": float,
#     },
# )
# swap_objects_grab_banana_subtask = CheckArgsToolCallSubTask(
#     expected_tool_name="move_to_point",
#     expected_args={
#         "x": BANANA_POSITION.x,
#         "y": BANANA_POSITION.y,
#         "z": BANANA_POSITION.z,
#         "task": "grab",
#     },
# )
# swap_objects_move_banana_to_cube_subtask = CheckArgsToolCallSubTask(
#     expected_tool_name="move_to_point",
#     expected_args={
#         "x": CUBE_POSITION.x,
#         "y": CUBE_POSITION.y,
#         "z": CUBE_POSITION.z,
#         "task": "drop",
#     },
# )
# swap_objects_grab_cube_from_temp_subtask = CheckArgsToolCallSubTask(
#     expected_tool_name="move_to_point",
#     expected_args={
#         "task": "grab",
#     },
#     expected_optional_args={
#         "x": float,
#         "y": float,
#         "z": float,
#     },
# )
# swap_objects_move_cube_to_banana_subtask = CheckArgsToolCallSubTask(
#     expected_tool_name="move_to_point",
#     expected_args={
#         "x": BANANA_POSITION.x,
#         "y": BANANA_POSITION.y,
#         "z": BANANA_POSITION.z,
#         "task": "drop",
#     },
# )

######### VALIDATORS #########################################################################################
move_to_point_ord_val_grab = OrderedCallsValidator(
    subtasks=[move_to_point_subtask_grab]
)
move_to_point_ord_val_drop = OrderedCallsValidator(
    subtasks=[move_to_point_subtask_drop]
)

get_both_object_positions_ord_val = NotOrderedCallsValidator(
    subtasks=[get_object_positions_cube_subtask, get_object_positions_banana_subtask]
)

grab_cube_ord_val = OrderedCallsValidator(
    subtasks=[get_object_positions_cube_subtask, grab_cube_move_subtask]
)
grab_banana_ord_val = OrderedCallsValidator(
    subtasks=[get_object_positions_banana_subtask, grab_banana_move_subtask]
)


move_cube_left_ord_val = OrderedCallsValidator(
    subtasks=[
        get_object_positions_cube_subtask,
        move_cube_left_grab_subtask,
        move_cube_left_drop_subtask,
    ]
)
move_banana_left_ord_val = OrderedCallsValidator(
    subtasks=[
        get_object_positions_banana_subtask,
        move_banana_left_grab_subtask,
        move_banana_left_drop_subtask,
    ]
)

move_cube_front_ord_val = OrderedCallsValidator(
    subtasks=[
        get_object_positions_cube_subtask,
        move_cube_front_grab_subtask,
        move_cube_front_drop_subtask,
    ]
)
move_banana_front_ord_val = OrderedCallsValidator(
    subtasks=[
        get_object_positions_banana_subtask,
        move_banana_front_grab_subtask,
        move_banana_front_drop_subtask,
    ]
)

# swap_objects_ord_val = OrderedCallsValidator(
#     subtasks=[
#         swap_objects_grab_cube_subtask,
#         swap_objects_move_cube_to_temp_subtask,
#         swap_objects_grab_banana_subtask,
#         swap_objects_move_banana_to_cube_subtask,
#         swap_objects_grab_cube_from_temp_subtask,
#         swap_objects_move_cube_to_banana_subtask,
#     ]
# )


def get_manipulation_tasks(
    extra_tool_calls: List[int] = [0],
    prompt_detail: List[Literal["brief", "descriptive"]] = ["brief", "descriptive"],
    n_shots: List[Literal[0, 2, 5]] = [0, 2, 5],
) -> List[Task]:
    """Get predefined manipulation tasks.

    Parameters
    ----------
    Parameters match :class:`~src.rai_bench.rai_bench.test_models.ToolCallingAgentBenchmarkConfig`.
    See the class documentation for parameter descriptions.

    Returns
    -------
    Returned list match :func:`~src.rai_bench.rai_bench.tool_calling_agent.predefined.tasks.get_tasks`.
    """
    tasks: List[Task] = []

    # Objects for tasks requiring single positions
    objects = {
        BANANA_OBJECT: [BANANA_POSITION],
        CUBE_OBJECT: [CUBE_POSITION],
    }

    # Objects for GetObjectPositionsTask with multiple banana positions
    objects_with_multiple_bananas = {
        BANANA_OBJECT: [BANANA_POSITION, BANANA_POSITION_2],
        CUBE_OBJECT: [CUBE_POSITION],
    }

    # Generate all combinations of prompt_detail and n_shots and extra tool calls
    for extra_calls in extra_tool_calls:
        for detail in prompt_detail:
            for shots in n_shots:
                task_args = TaskArgs(
                    extra_tool_calls=extra_calls,
                    prompt_detail=detail,
                    examples_in_system_prompt=shots,
                )

                tasks.extend(
                    [
                        MoveToPointTask(
                            objects=objects,
                            move_to_tool_input=MoveToPointToolInput(
                                x=1.0, y=2.0, z=3.0, task="grab"
                            ),
                            validators=[move_to_point_ord_val_grab],
                            task_args=task_args,
                        ),
                        MoveToPointTask(
                            objects=objects,
                            move_to_tool_input=MoveToPointToolInput(
                                x=1.2, y=2.3, z=3.4, task="drop"
                            ),
                            validators=[move_to_point_ord_val_drop],
                            task_args=task_args,
                        ),
                        GetObjectPositionsTask(
                            objects=objects_with_multiple_bananas,
                            validators=[get_both_object_positions_ord_val],
                            task_args=task_args,
                        ),
                        GrabExistingObjectTask(
                            objects=objects,
                            object_to_grab=CUBE_OBJECT,
                            validators=[grab_cube_ord_val],
                            task_args=task_args,
                        ),
                        GrabExistingObjectTask(
                            objects=objects,
                            object_to_grab=BANANA_OBJECT,
                            validators=[grab_banana_ord_val],
                            task_args=task_args,
                        ),
                        MoveExistingObjectLeftTask(
                            objects=objects,
                            object_to_grab=CUBE_OBJECT,
                            validators=[move_cube_left_ord_val],
                            task_args=task_args,
                        ),
                        MoveExistingObjectLeftTask(
                            objects=objects,
                            object_to_grab=BANANA_OBJECT,
                            validators=[move_banana_left_ord_val],
                            task_args=task_args,
                        ),
                        MoveExistingObjectFrontTask(
                            objects=objects,
                            object_to_grab=CUBE_OBJECT,
                            validators=[move_cube_front_ord_val],
                            task_args=task_args,
                        ),
                        MoveExistingObjectFrontTask(
                            objects=objects,
                            object_to_grab=BANANA_OBJECT,
                            validators=[move_banana_front_ord_val],
                            task_args=task_args,
                        ),
                        # SwapObjectsTask(
                        #     objects=objects,
                        #     objects_to_swap=[CUBE_OBJECT, BANANA_OBJECT],
                        #     validators=[swap_objects_ord_val],
                        #     task_args=task_args,
                        # ),
                    ]
                )

    return tasks
