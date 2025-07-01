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
    CheckActionFieldsToolCallSubTask,
)
from rai_bench.tool_calling_agent.tasks.navigation import (
    MoveToBedTask,
    MoveToFrontTask,
    NavigateToPointTask,
    SpinAroundTask,
)
from rai_bench.tool_calling_agent.validators import (
    OrderedCallsValidator,
)

########## SUBTASKS #################################################################

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
######### VALIDATORS #########################################################################################
start_navigate_action_ord_val = OrderedCallsValidator(
    subtasks=[start_nav_action_subtask]
)
start_spin_action_ord_val = OrderedCallsValidator(subtasks=[start_spin_action_subtask])
move_ahead_ord_val = OrderedCallsValidator(subtasks=[start_move_front_action_subtask])


def get_navigation_tasks(
    extra_tool_calls: List[int] = [0],
    prompt_detail: List[Literal["brief", "descriptive"]] = ["brief", "descriptive"],
    n_shots: List[Literal[0, 2, 5]] = [0, 2, 5],
) -> List[Task]:
    """Get predefined navigation tasks.

    Parameters
    ----------
    Parameters match :class:`~src.rai_bench.rai_bench.test_models.ToolCallingAgentBenchmarkConfig`.
    See the class documentation for parameter descriptions.

    Returns
    -------
    Returned list match :func:`~src.rai_bench.rai_bench.tool_calling_agent.predefined.tasks.get_tasks`.
    """
    tasks: List[Task] = []

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
                        NavigateToPointTask(
                            validators=[start_navigate_action_ord_val],
                            task_args=task_args,
                        ),
                        SpinAroundTask(
                            validators=[start_spin_action_ord_val],
                            task_args=task_args,
                        ),
                        MoveToBedTask(
                            validators=[move_ahead_ord_val],
                            task_args=task_args,
                        ),
                        MoveToFrontTask(
                            validators=[move_ahead_ord_val],
                            task_args=task_args,
                        ),
                    ]
                )

    return tasks
