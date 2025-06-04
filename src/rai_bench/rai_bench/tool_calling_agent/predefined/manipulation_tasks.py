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

from typing import List, Literal

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
    MoveToPointTask,
)
from rai_bench.tool_calling_agent.validators import (
    OrderedCallsValidator,
)

########## SUBTASKS #################################################################
move_to_point_subtask_grab = CheckArgsToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args={"x": 1.0, "y": 2.0, "z": 3.0, "task": "grab"},
)
move_to_point_subtask_drop = CheckArgsToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args={"x": 1.2, "y": 2.3, "z": 3.4, "task": "drop"},
)

######### VALIDATORS #########################################################################################
move_to_point_ord_val_grab = OrderedCallsValidator(
    subtasks=[move_to_point_subtask_grab]
)
move_to_point_ord_val_drop = OrderedCallsValidator(
    subtasks=[move_to_point_subtask_drop]
)


def get_manipulation_tasks(
    extra_tool_calls: List[int] = [0],
    prompt_detail: List[Literal["brief", "moderate", "descriptive"]] = [
        "brief",
        "moderate",
        "descriptive",
    ],
    n_shots: List[Literal[0, 2, 5]] = [0, 2, 5],
) -> List[Task]:
    tasks: List[Task] = []

    objects = {
        "banana": [Point(x=0.1, y=0.2, z=0.3), Point(x=0.4, y=0.5, z=0.6)],
        "cube": [Point(x=0.7, y=0.8, z=0.9)],
    }
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
                    ]
                )

    return tasks
