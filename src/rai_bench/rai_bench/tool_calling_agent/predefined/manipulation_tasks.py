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
from rai_bench.tool_calling_agent.tasks.manipulation import (
    AlignTwoObjectsTask,
    GetObjectPositionsTask,
    GrabExistingObjectTask,
    MoveExistingObjectFrontTask,
    MoveExistingObjectLeftTask,
    MoveToPointTask,
)

BANANA_POSITION = Point(x=0.1, y=0.2, z=0.3)
BANANA_POSITION_2 = Point(x=0.4, y=0.5, z=0.6)
CUBE_POSITION = Point(x=0.7, y=0.8, z=0.9)

BANANA_OBJECT = "banana"
CUBE_OBJECT = "cube"
APPLE_OBJECT = "apple"

MOVE_TO_GRAB_COORDS: Dict[str, Any] = {"x": 1.0, "y": 2.0, "z": 3.0, "task": "grab"}
MOVE_TO_DROP_COORDS: Dict[str, Any] = {"x": 1.2, "y": 2.3, "z": 3.4, "task": "drop"}


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

    objects = {
        BANANA_OBJECT: [BANANA_POSITION],
        CUBE_OBJECT: [CUBE_POSITION],
    }

    objects_with_multiple_bananas = {
        BANANA_OBJECT: [BANANA_POSITION, BANANA_POSITION_2],
        CUBE_OBJECT: [CUBE_POSITION],
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
                            task_args=task_args,
                        ),
                        MoveToPointTask(
                            objects=objects,
                            move_to_tool_input=MoveToPointToolInput(
                                x=1.2, y=2.3, z=3.4, task="drop"
                            ),
                            task_args=task_args,
                        ),
                        GetObjectPositionsTask(
                            objects=objects_with_multiple_bananas,
                            task_args=task_args,
                        ),
                        GrabExistingObjectTask(
                            objects=objects,
                            object_to_grab=CUBE_OBJECT,
                            task_args=task_args,
                        ),
                        GrabExistingObjectTask(
                            objects=objects,
                            object_to_grab=BANANA_OBJECT,
                            task_args=task_args,
                        ),
                        MoveExistingObjectLeftTask(
                            objects=objects,
                            object_to_grab=CUBE_OBJECT,
                            task_args=task_args,
                        ),
                        MoveExistingObjectLeftTask(
                            objects=objects,
                            object_to_grab=BANANA_OBJECT,
                            task_args=task_args,
                        ),
                        MoveExistingObjectFrontTask(
                            objects=objects,
                            object_to_grab=CUBE_OBJECT,
                            task_args=task_args,
                        ),
                        MoveExistingObjectFrontTask(
                            objects=objects,
                            object_to_grab=BANANA_OBJECT,
                            task_args=task_args,
                        ),
                        AlignTwoObjectsTask(objects=objects, task_args=task_args),
                    ]
                )

    return tasks
