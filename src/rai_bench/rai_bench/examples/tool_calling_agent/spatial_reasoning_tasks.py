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

from rai_bench.tool_calling_agent.tasks.navigation import (
    MoveToBedTask,
    MoveToFrontTask,
    NavigateToPointTask,
    NavigationTask,
    SpinAroundTask,
)
from rai_bench.tool_calling_agent.tasks.spatial import (
    BoolImageTask,
    BoolImageTaskInput,
    SpatialReasoningAgentTask,
)
from rai_bench.tool_calling_agent.tasks.subtasks import (
    CheckActionFieldsToolCallSubTask,
    CheckArgsToolCallSubTask,
)
from rai_bench.tool_calling_agent.validators import OrderedCallsValidator

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

start_nav_action_subtask = CheckActionFieldsToolCallSubTask(
    expected_tool_name="start_ros2_action",
    expected_action="/navigate_to_pose",
    expected_action_type="nav2_msgs/action/NavigateToPose",
    expected_fields={
        "pose": {
            "header": {"frame_id": "map"},
            "pose": {
                "position": {"x": 2.0, "y": 2.0, "z": 0.0},
            },
        },
    },
)
start_spin_action_subtask = CheckActionFieldsToolCallSubTask(
    expected_tool_name="start_ros2_action",
    expected_action="/spin",
    expected_action_type="nav2_msgs/action/Spin",
    expected_fields={"target_yaw": 3},
)
start_move_front_action_subtask = CheckActionFieldsToolCallSubTask(
    expected_tool_name="start_ros2_action",
    expected_action="/drive_on_heading",
    expected_action_type="nav2_msgs/action/DriveOnHeading",
    expected_fields={
        "target": {"y": 0.0, "z": 0.0},
    },
)
########## validators
ret_true_ord_val = OrderedCallsValidator(subtasks=[return_true_subtask])
ret_false_ord_val = OrderedCallsValidator(subtasks=[return_false_subtask])

start_navigate_action_ord_val = OrderedCallsValidator(
    subtasks=[start_nav_action_subtask]
)
start_spin_action_ord_val = OrderedCallsValidator(subtasks=[start_spin_action_subtask])
move_ahead_ord_val = OrderedCallsValidator(subtasks=[start_move_front_action_subtask])
########## tasks
true_tasks: Sequence[SpatialReasoningAgentTask] = [
    BoolImageTask(
        task_input=input_item, validators=[ret_true_ord_val], extra_tool_calls=0
    )
    for input_item in true_response_inputs
]
false_tasks: Sequence[SpatialReasoningAgentTask] = [
    BoolImageTask(task_input=input_item, validators=[ret_false_ord_val])
    for input_item in false_response_inputs
]

nav_tasks: Sequence[NavigationTask] = [
    NavigateToPointTask(validators=[start_navigate_action_ord_val], extra_tool_calls=5),
    SpinAroundTask(validators=[start_spin_action_ord_val], extra_tool_calls=5),
    MoveToBedTask(validators=[move_ahead_ord_val], extra_tool_calls=5),
    MoveToFrontTask(validators=[move_ahead_ord_val], extra_tool_calls=5),
]


all_tasks = nav_tasks
random.shuffle(all_tasks)
