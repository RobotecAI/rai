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

from typing import List, Literal, Sequence

from rai_bench.tool_calling_agent.interfaces import (
    Task,
    TaskArgs,
)
from rai_bench.tool_calling_agent.subtasks import (
    CheckArgsToolCallSubTask,
)
from rai_bench.tool_calling_agent.tasks.spatial import (
    BoolImageTaskEasy,
    BoolImageTaskHard,
    BoolImageTaskInput,
    BoolImageTaskMedium,
)
from rai_bench.tool_calling_agent.validators import (
    OrderedCallsValidator,
)

IMG_PATH = "src/rai_bench/rai_bench/tool_calling_agent/predefined/images/"
true_response_inputs: List[BoolImageTaskInput] = [
    BoolImageTaskInput(
        question="Is the door on the left from the desk?",
        images_paths=[IMG_PATH + "image_1.jpg"],
    ),
    BoolImageTaskInput(
        question="Is the light on in the room?",
        images_paths=[IMG_PATH + "image_2.jpg"],
    ),
    BoolImageTaskInput(
        question="Do you see the plant?",
        images_paths=[IMG_PATH + "image_2.jpg"],
    ),
    BoolImageTaskInput(
        question="Are there any pictures on the wall?",
        images_paths=[IMG_PATH + "image_3.jpg"],
    ),
    BoolImageTaskInput(
        question="Are there 3 pictures on the wall?",
        images_paths=[IMG_PATH + "image_4.jpg"],
    ),
    BoolImageTaskInput(
        question="Is there a plant behind the rack?",
        images_paths=[IMG_PATH + "image_5.jpg"],
    ),
    BoolImageTaskInput(
        question="Is there a pillow on the armchain?",
        images_paths=[IMG_PATH + "image_7.jpg"],
    ),
]
false_response_inputs: List[BoolImageTaskInput] = [
    BoolImageTaskInput(
        question="Is the door open?",
        images_paths=[IMG_PATH + "image_1.jpg"],
    ),
    BoolImageTaskInput(
        question="Is someone in the room?",
        images_paths=[IMG_PATH + "image_1.jpg"],
    ),
    BoolImageTaskInput(
        question="Do you see the plant?",
        images_paths=[IMG_PATH + "image_3.jpg"],
    ),
    BoolImageTaskInput(
        question="Are there 4 pictures on the wall?",
        images_paths=[IMG_PATH + "image_4.jpg"],
    ),
    BoolImageTaskInput(
        question="Is there a rack on the left from the sofa?",
        images_paths=[IMG_PATH + "image_4.jpg"],
    ),
    BoolImageTaskInput(
        question="Is there a plant on the right from the window?",
        images_paths=[IMG_PATH + "image_6.jpg"],
    ),
    BoolImageTaskInput(
        question="Is there a red pillow on the armchair?",
        images_paths=[IMG_PATH + "image_7.jpg"],
    ),
]
########## SUBTASKS #################################################################
return_true_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="return_bool_response", expected_args={"response": True}
)
return_false_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="return_bool_response", expected_args={"response": False}
)

######### VALIDATORS #########################################################################################
ret_true_ord_val = OrderedCallsValidator(subtasks=[return_true_subtask])
ret_false_ord_val = OrderedCallsValidator(subtasks=[return_false_subtask])


def get_spatial_tasks(
    extra_tool_calls: List[int] = [0],
    prompt_detail: List[Literal["brief", "descriptive"]] = ["brief", "descriptive"],
    n_shots: List[Literal[0, 2, 5]] = [0, 2, 5],
) -> Sequence[Task]:
    """Get predefined spatial reasoning tasks.

    Parameters
    ----------
    extra_tool_calls : List[int], optional
        how many extra tool calls allowed to still pass, by default [0]
    prompt_detail : List[Literal["brief", "descriptive"]], optional
        how descriptive should task prompt be, by default all levels are included:
        ["brief", "descriptive"]
    n_shots : List[Literal[0, 2, 5]], optional
        how many examples are in system prompt, by default all are included: [0, 2, 5]

    Returns
    -------
    Sequence[Task]
        sequence of spatial reasoning tasks with varying difficulty levels.
        There will be len(extra_tool_calls) * len(prompt_detail) * len(n_shots) tasks generated.
    """
    tasks: List[Task] = []

    # Categorize tasks by complexity based on question difficulty
    easy_true_inputs = [
        # Single object presence/detection
        BoolImageTaskInput(
            question="Is the light on in the room?",
            images_paths=[IMG_PATH + "image_2.jpg"],
        ),
        BoolImageTaskInput(
            question="Do you see the plant?", images_paths=[IMG_PATH + "image_2.jpg"]
        ),
        BoolImageTaskInput(
            question="Are there any pictures on the wall?",
            images_paths=[IMG_PATH + "image_3.jpg"],
        ),
        BoolImageTaskInput(
            question="Is there a pillow on the armchain?",
            images_paths=[IMG_PATH + "image_7.jpg"],
        ),
    ]

    medium_true_inputs = [
        # Object state or counting
        BoolImageTaskInput(
            question="Are there 3 pictures on the wall?",
            images_paths=[IMG_PATH + "image_4.jpg"],
        ),
    ]

    hard_true_inputs = [
        # Spatial relationships between objects
        BoolImageTaskInput(
            question="Is the door on the left from the desk?",
            images_paths=[IMG_PATH + "image_1.jpg"],
        ),
        BoolImageTaskInput(
            question="Is there a plant behind the rack?",
            images_paths=[IMG_PATH + "image_5.jpg"],
        ),
    ]

    easy_false_inputs = [
        # Single object presence/detection
        BoolImageTaskInput(
            question="Is someone in the room?", images_paths=[IMG_PATH + "image_1.jpg"]
        ),
        BoolImageTaskInput(
            question="Do you see the plant?", images_paths=[IMG_PATH + "image_3.jpg"]
        ),
        BoolImageTaskInput(
            question="Is there a red pillow on the armchair?",
            images_paths=[IMG_PATH + "image_7.jpg"],
        ),
    ]

    medium_false_inputs = [
        # Object state or counting
        BoolImageTaskInput(
            question="Is the door open?", images_paths=[IMG_PATH + "image_1.jpg"]
        ),
        BoolImageTaskInput(
            question="Are there 4 pictures on the wall?",
            images_paths=[IMG_PATH + "image_4.jpg"],
        ),
    ]

    hard_false_inputs = [
        # Spatial relationships between objects
        BoolImageTaskInput(
            question="Is there a rack on the left from the sofa?",
            images_paths=[IMG_PATH + "image_4.jpg"],
        ),
        BoolImageTaskInput(
            question="Is there a plant on the right from the window?",
            images_paths=[IMG_PATH + "image_6.jpg"],
        ),
    ]

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
                        BoolImageTaskEasy(
                            task_input=input_item,
                            validators=[ret_true_ord_val],
                            task_args=task_args,
                        )
                        for input_item in easy_true_inputs
                    ]
                )

                tasks.extend(
                    [
                        BoolImageTaskEasy(
                            task_input=input_item,
                            validators=[ret_false_ord_val],
                            task_args=task_args,
                        )
                        for input_item in easy_false_inputs
                    ]
                )

                tasks.extend(
                    [
                        BoolImageTaskMedium(
                            task_input=input_item,
                            validators=[ret_true_ord_val],
                            task_args=task_args,
                        )
                        for input_item in medium_true_inputs
                    ]
                )

                tasks.extend(
                    [
                        BoolImageTaskMedium(
                            task_input=input_item,
                            validators=[ret_false_ord_val],
                            task_args=task_args,
                        )
                        for input_item in medium_false_inputs
                    ]
                )

                tasks.extend(
                    [
                        BoolImageTaskHard(
                            task_input=input_item,
                            validators=[ret_true_ord_val],
                            task_args=task_args,
                        )
                        for input_item in hard_true_inputs
                    ]
                )

                tasks.extend(
                    [
                        BoolImageTaskHard(
                            task_input=input_item,
                            validators=[ret_false_ord_val],
                            task_args=task_args,
                        )
                        for input_item in hard_false_inputs
                    ]
                )

    return tasks
