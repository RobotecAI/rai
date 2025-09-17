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
from rai_bench.tool_calling_agent.tasks.custom_interfaces import (
    CallGetLogDigestTask,
    CallGetLogDigestTaskIndirect,
    CallGroundedSAMSegmentTask,
    CallGroundedSAMSegmentTaskIndirect,
    CallGroundingDinoClassify,
    CallGroundingDinoClassifyIndirect,
    CallROS2ManipulatorMoveToServiceTask,
    CallROS2ManipulatorMoveToServiceTaskIndirect,
    CallVectorStoreRetrievalTask,
    CallVectorStoreRetrievalTaskIndirect,
    CompleteObjectInteractionTask,
    CompleteObjectInteractionTaskIndirect,
    EmergencyResponseProtocolTask,
    EmergencyResponseProtocolTaskIndirect,
    MultiModalSceneDocumentationTask,
    MultiModalSceneDocumentationTaskIndirect,
    PublishROS2AudioMessageTask,
    PublishROS2AudioMessageTaskIndirect,
    PublishROS2DetectionArrayTask,
    PublishROS2DetectionArrayTaskIndirect,
    PublishROS2HRIMessageTextTask,
    PublishROS2HRIMessageTextTaskIndirect,
)

# Object Classes
PERSON_CLASS = "person"
BOTTLE_CLASS = "bottle"

# Text Messages
HRI_TEXT = "Hello!"

# Audio Parameters
BASIC_AUDIO_SAMPLES = [123, 456, 789]
BASIC_SAMPLE_RATE = 44100
BASIC_CHANNELS = 2

# Position Parameters
STANDARD_TARGET_POSITION = (1.0, 2.0, 3.0)

# Query Strings
ROBOT_PURPOSE_QUERY = "What is the purpose of this robot?"
GROUNDING_DINO_CLASSES = "person, bottle"

# Default scene objects for documentation
DEFAULT_SCENE_OBJECTS = ["person", "bottle"]


def get_custom_interfaces_tasks(
    extra_tool_calls: List[int] = [0],
    prompt_detail: List[Literal["brief", "descriptive"]] = ["brief", "descriptive"],
    n_shots: List[Literal[0, 2, 5]] = [0, 2, 5],
    include_indirect: bool = True,
) -> List[Task]:
    """Get predefined custom_interfaces tasks.

    Parameters
    ----------
    extra_tool_calls : List[int]
        Number of extra tool calls allowed beyond the minimum required.
    prompt_detail : List[Literal["brief", "descriptive"]]
        Level of detail in task prompts.
    n_shots : List[Literal[0, 2, 5]]
        Number of examples in system prompt.
    include_indirect : bool
        Whether to include indirect (natural language) task variants.

    Returns
    -------
    List[Task]
        List of task instances for benchmarking.
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

                # Direct tasks (original)
                tasks.append(
                    PublishROS2HRIMessageTextTask(
                        task_args=task_args,
                        text=HRI_TEXT,
                    ),
                )
                tasks.append(
                    PublishROS2AudioMessageTask(
                        task_args=task_args,
                        audio=BASIC_AUDIO_SAMPLES,
                        sample_rate=BASIC_SAMPLE_RATE,
                        channels=BASIC_CHANNELS,
                    )
                )

                tasks.append(
                    PublishROS2DetectionArrayTask(
                        task_args=task_args,
                        detection_classes=[PERSON_CLASS],
                    )
                )
                tasks.append(
                    PublishROS2DetectionArrayTask(
                        task_args=task_args,
                        detection_classes=[BOTTLE_CLASS, PERSON_CLASS],
                    )
                )

                tasks.append(
                    CallROS2ManipulatorMoveToServiceTask(
                        task_args=task_args,
                        target_x=STANDARD_TARGET_POSITION[0],
                        target_y=STANDARD_TARGET_POSITION[1],
                        target_z=STANDARD_TARGET_POSITION[2],
                        initial_gripper_state=True,
                        final_gripper_state=False,
                    )
                )

                tasks.append(
                    CallGroundedSAMSegmentTask(
                        task_args=task_args,
                        detection_classes=[BOTTLE_CLASS],
                    )
                )

                tasks.append(
                    CallGroundingDinoClassify(
                        task_args=task_args,
                        classes=GROUNDING_DINO_CLASSES,
                    )
                )

                tasks.append(
                    CallGetLogDigestTask(
                        task_args=task_args,
                    )
                )
                tasks.append(
                    CallVectorStoreRetrievalTask(
                        task_args=task_args,
                        query=ROBOT_PURPOSE_QUERY,
                    )
                )

                tasks.append(
                    CompleteObjectInteractionTask(
                        task_args=task_args,
                        target_class=BOTTLE_CLASS,
                    )
                )

                tasks.append(
                    MultiModalSceneDocumentationTask(
                        task_args=task_args,
                        objects=DEFAULT_SCENE_OBJECTS,
                    )
                )

                tasks.append(
                    EmergencyResponseProtocolTask(
                        task_args=task_args,
                        target_class=PERSON_CLASS,
                    )
                )

                if include_indirect:
                    tasks.append(
                        PublishROS2HRIMessageTextTaskIndirect(
                            task_args=task_args,
                            text=HRI_TEXT,
                        ),
                    )
                    tasks.append(
                        PublishROS2AudioMessageTaskIndirect(
                            task_args=task_args,
                            audio=BASIC_AUDIO_SAMPLES,
                            sample_rate=BASIC_SAMPLE_RATE,
                            channels=BASIC_CHANNELS,
                        )
                    )

                    tasks.append(
                        PublishROS2DetectionArrayTaskIndirect(
                            task_args=task_args,
                            detection_classes=[PERSON_CLASS],
                        )
                    )
                    tasks.append(
                        PublishROS2DetectionArrayTaskIndirect(
                            task_args=task_args,
                            detection_classes=[BOTTLE_CLASS, PERSON_CLASS],
                        )
                    )

                    tasks.append(
                        CallROS2ManipulatorMoveToServiceTaskIndirect(
                            task_args=task_args,
                            target_x=STANDARD_TARGET_POSITION[0],
                            target_y=STANDARD_TARGET_POSITION[1],
                            target_z=STANDARD_TARGET_POSITION[2],
                            initial_gripper_state=True,
                            final_gripper_state=False,
                        )
                    )

                    tasks.append(
                        CallGroundedSAMSegmentTaskIndirect(
                            task_args=task_args,
                            detection_classes=[BOTTLE_CLASS],
                        )
                    )

                    tasks.append(
                        CallGroundingDinoClassifyIndirect(
                            task_args=task_args,
                            classes=GROUNDING_DINO_CLASSES,
                        )
                    )

                    tasks.append(
                        CallGetLogDigestTaskIndirect(
                            task_args=task_args,
                        )
                    )
                    tasks.append(
                        CallVectorStoreRetrievalTaskIndirect(
                            task_args=task_args,
                            query=ROBOT_PURPOSE_QUERY,
                        )
                    )

                    tasks.append(
                        CompleteObjectInteractionTaskIndirect(
                            task_args=task_args,
                            target_class=BOTTLE_CLASS,
                        )
                    )

                    tasks.append(
                        MultiModalSceneDocumentationTaskIndirect(
                            task_args=task_args,
                            objects=DEFAULT_SCENE_OBJECTS,
                        )
                    )

                    tasks.append(
                        EmergencyResponseProtocolTaskIndirect(
                            task_args=task_args,
                            target_class=PERSON_CLASS,
                        )
                    )

    return tasks
