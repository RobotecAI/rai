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

import random
from typing import List, Sequence

from rai_bench.tool_calling_agent.tasks.spatial import (
    BoolImageTask,
    BoolImageTaskInput,
    SpatialReasoningAgentTask,
)

from ...tool_calling_agent.tasks.subtasks import CheckArgsToolCallSubTask
from ...tool_calling_agent.validators import OrderedCallsValidator

true_response_inputs: List[BoolImageTaskInput] = [
    BoolImageTaskInput(
        question="Is the door on the left from the desk?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_1.jpg"],
    ),
    BoolImageTaskInput(
        question="Is the light on in the room?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_2.jpg"],
    ),
    BoolImageTaskInput(
        question="Do you see the plant?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_2.jpg"],
    ),
    BoolImageTaskInput(
        question="Are there any pictures on the wall?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_3.jpg"],
    ),
    BoolImageTaskInput(
        question="Are there 3 pictures on the wall?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_4.jpg"],
    ),
    BoolImageTaskInput(
        question="Is there a plant behind the rack?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_5.jpg"],
    ),
    BoolImageTaskInput(
        question="Is there a pillow on the armchain?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_7.jpg"],
    ),
]
false_response_inputs: List[BoolImageTaskInput] = [
    BoolImageTaskInput(
        question="Is the door open?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_1.jpg"],
    ),
    BoolImageTaskInput(
        question="Is someone in the room?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_1.jpg"],
    ),
    BoolImageTaskInput(
        question="Do you see the plant?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_3.jpg"],
    ),
    BoolImageTaskInput(
        question="Are there 4 pictures on the wall?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_4.jpg"],
    ),
    BoolImageTaskInput(
        question="Is there a rack on the left from the sofa?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_4.jpg"],
    ),
    BoolImageTaskInput(
        question="Is there a plant on the right from the window?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_6.jpg"],
    ),
    BoolImageTaskInput(
        question="Is there a red pillow on the armchair?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_7.jpg"],
    ),
]
########## subtasks
return_true_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="return_bool_response", expected_args={"response": True}
)
return_false_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="return_bool_response", expected_args={"response": False}
)
########## validators
ret_true_ord_val = OrderedCallsValidator(subtasks=[return_true_subtask])
ret_false_ord_val = OrderedCallsValidator(subtasks=[return_false_subtask])

########## tasks
true_tasks: Sequence[SpatialReasoningAgentTask] = [
    BoolImageTask(
        task_input=input_item, validators=[ret_true_ord_val], extra_tool_calls=0
    )
    for input_item in true_response_inputs
]
false_tasks: Sequence[SpatialReasoningAgentTask] = [
    BoolImageTask(
        task_input=input_item, validators=[ret_false_ord_val], extra_tool_calls=0
    )
    for input_item in false_response_inputs
]

all_tasks = true_tasks + false_tasks
random.shuffle(all_tasks)
