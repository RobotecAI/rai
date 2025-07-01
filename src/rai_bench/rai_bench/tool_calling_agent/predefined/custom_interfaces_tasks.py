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

from rai_bench.tool_calling_agent.interfaces import (
    Task,
    TaskArgs,
)
from rai_bench.tool_calling_agent.subtasks import (
    CheckArgsToolCallSubTask,
    CheckTopicFieldsToolCallSubTask,
)
from rai_bench.tool_calling_agent.tasks.custom_interfaces import (
    PublishROS2HRIMessageTextTask,
)
from rai_bench.tool_calling_agent.validators import (
    OrderedCallsValidator,
)

########## SUBTASKS #################################################################
pub_HRIMessage_text_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic="/to_human",
    expected_message_type="rai_interfaces/msg/HRIMessage",
    expected_fields={"text": "Hello!"},
)

get_HRIMessage_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": "rai_interfaces/msg/HRIMessage"},
)


######### VALIDATORS #########################################################################################
pub_HRIMessage_text_ord_val = OrderedCallsValidator(
    subtasks=[pub_HRIMessage_text_subtask]
)
get_interface_publish_ord_val = OrderedCallsValidator(
    subtasks=[
        get_HRIMessage_interface_subtask,
        pub_HRIMessage_text_subtask,
    ]
)


def get_custom_interfaces_tasks(
    extra_tool_calls: List[int] = [0],
    prompt_detail: List[Literal["brief", "descriptive"]] = ["brief", "descriptive"],
    n_shots: List[Literal[0, 2, 5]] = [0, 2, 5],
) -> List[Task]:
    tasks: List[Task] = []

    for extra_calls in extra_tool_calls:
        for detail in prompt_detail:
            for shots in n_shots:
                task_args = TaskArgs(
                    extra_tool_calls=extra_calls,
                    prompt_detail=detail,
                    examples_in_system_prompt=shots,
                )
                tasks.append(
                    PublishROS2HRIMessageTextTask(
                        topic="/to_human",
                        validators=[
                            get_interface_publish_ord_val,
                        ],
                        task_args=task_args,
                        text="Hello!",
                    ),
                )

    return tasks
