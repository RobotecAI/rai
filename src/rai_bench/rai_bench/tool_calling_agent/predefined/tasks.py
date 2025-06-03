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
from typing import List, Literal

from rai_bench.tool_calling_agent.interfaces import (
    Task,
)
from rai_bench.tool_calling_agent.predefined import (
    get_basic_tasks,
    get_custom_interfaces_tasks,
    get_manipulation_tasks,
    get_navigation_tasks,
    get_spatial_tasks,
)


def get_tasks(
    extra_tool_calls: int = 0,
    complexities: List[Literal["easy", "medium", "hard"]] = ["easy", "medium", "hard"],
    prompt_detail: List[Literal["brief", "moderate", "descriptive"]] = [
        "brief",
        "moderate",
        "descriptive",
    ],
    n_shots: List[Literal[0, 2, 5]] = [0, 2, 5],
    task_types: List[
        Literal[
            "basic",
            "manipulation",
            "navigation",
            "custom_interfaces",
            "spatial_reasoning",
        ]
    ] = [
        "basic",
        "manipulation",
        "navigation",
        "custom_interfaces",
        "spatial_reasoning",
    ],
) -> List[Task]:
    all_tasks: List[Task] = []
    if "basic" in task_types:
        all_tasks += get_basic_tasks(
            extra_tool_calls=extra_tool_calls,
            prompt_detail=prompt_detail,
            n_shots=n_shots,
        )
    if "custom_interfaces" in task_types:
        all_tasks += get_custom_interfaces_tasks(extra_tool_calls=extra_tool_calls)
    if "manipulation" in task_types:
        all_tasks += get_manipulation_tasks(extra_tool_calls=extra_tool_calls)
    if "navigation" in task_types:
        all_tasks += get_navigation_tasks(extra_tool_calls=extra_tool_calls)
    if "spatial_reasoning" in task_types:
        all_tasks += get_spatial_tasks(extra_tool_calls=extra_tool_calls)

    filtered_tasks: List[Task] = []
    for task in all_tasks:
        if task.complexity not in complexities:
            continue

        filtered_tasks.append(task)

    random.shuffle(all_tasks)
    return all_tasks
