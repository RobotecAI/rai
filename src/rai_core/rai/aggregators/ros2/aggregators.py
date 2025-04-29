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

from typing import Any, List, cast

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from rcl_interfaces.msg import Log
from sensor_msgs.msg import CompressedImage, Image

from rai.aggregators import BaseAggregator
from rai.communication.ros2.api import convert_ros_img_to_base64
from rai.initialization.model_initialization import get_llm_model
from rai.messages import HumanMultimodalMessage


class ROS2LogsAggregator(BaseAggregator[Log]):
    """Returns only unique messages while keeping their order"""

    levels = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "FATAL"}

    def get(self) -> HumanMessage:
        msgs = self.get_buffer()
        buffer = []
        prev_parsed = None
        counter = 0
        for log in msgs:
            level = self.levels[log.level]
            parsed = f"[{log.name}] [{level}] [{log.function}] {log.msg}"
            if parsed == prev_parsed:
                counter += 1
                continue
            else:
                if counter != 0:
                    parsed = f"Log above repeated {counter} times"
            buffer.append(parsed)
            counter = 0
            prev_parsed = parsed
        result = f"Logs summary: {list(dict.fromkeys(buffer))}"
        self.clear_buffer()
        return HumanMessage(content=result)


class ROS2GetLastImageAggregator(BaseAggregator[Image | CompressedImage]):
    """Returns the last image from the buffer as base64 encoded string"""

    def get(self) -> HumanMultimodalMessage | None:
        msgs = self.get_buffer()
        if len(msgs) == 0:
            return None
        ros2_img = msgs[-1]
        b64_image = convert_ros_img_to_base64(ros2_img)
        self.clear_buffer()
        return HumanMultimodalMessage(content="", images=[b64_image])


class ROS2ImgVLMDescriptionAggregator(BaseAggregator[Image | CompressedImage]):
    """
    Returns the VLM analysis of the last image in the aggregation buffer
    """

    SYSTEM_PROMPT = "You are an expert in image analysis and your speciality is the description of images"

    def __init__(
        self, max_size: int | None = None, llm: BaseChatModel | None = None
    ) -> None:
        super().__init__(max_size)
        if llm is None:
            self.llm = get_llm_model(model_type="simple_model", streaming=True)
        else:
            self.llm = llm

    def get(self) -> HumanMessage | None:
        msgs: List[Image | CompressedImage] = self.get_buffer()
        if len(msgs) == 0:
            return None

        b64_images: List[str] = [convert_ros_img_to_base64(msg) for msg in msgs]
        self.clear_buffer()

        class ROS2ImgDescription(BaseModel):
            key_elements: List[str] = Field(
                ..., description="Key elements of the image"
            )

        task = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMultimodalMessage(
                content="Describe key elements that are currently in robot's view",
                images=[b64_images[-1]],
            ),
        ]
        llm = self.llm.with_structured_output(ROS2ImgDescription)
        response = cast(ROS2ImgDescription, llm.invoke(task))
        return HumanMessage(
            content=f"These are the key elements of the last camera image frame: {response}"
        )


class ROS2ImgVLMDiffAggregator(BaseAggregator[Image | CompressedImage]):
    """
    Returns the LLM analysis of the differences between 3 images in the
    aggregation buffer: 1st, midden, last
    """

    SYSTEM_PROMPT = "You are an expert in image analysis and your speciality is the comparison of 3 images"

    def __init__(
        self, max_size: int | None = None, llm: BaseChatModel | None = None
    ) -> None:
        super().__init__(max_size)
        if llm is None:
            self.llm = get_llm_model(model_type="simple_model", streaming=True)
        else:
            self.llm = llm

    @staticmethod
    def get_key_elements(elements: List[Any]) -> List[Any]:
        """
        Returns 1st, last and middle elements of the list
        """
        if len(elements) <= 3:
            return elements
        middle_index = len(elements) // 2
        return [elements[0], elements[middle_index], elements[-1]]

    def get(self) -> HumanMessage | None:
        msgs = self.get_buffer()
        if len(msgs) == 0:
            return None

        b64_images = [convert_ros_img_to_base64(msg) for msg in msgs]

        self.clear_buffer()

        b64_images = self.get_key_elements(b64_images)

        class ROS2ImgDiffOutput(BaseModel):
            are_different: bool = Field(
                ..., description="Whether the images are different"
            )
            differences: List[str] = Field(
                ..., description="Description of the difference"
            )

        task = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMultimodalMessage(
                content="Here are max 3 subsequent images from the robot camera. Robot might be moving. Outline key differences in robot's view.",
                images=b64_images,
            ),
        ]
        llm = self.llm.with_structured_output(ROS2ImgDiffOutput)
        response = cast(ROS2ImgDiffOutput, llm.invoke(task))
        return HumanMessage(
            content=f"Result of the analysis of the {len(b64_images)} keyframes selected from {len(b64_images)} last images:\n{response}"
        )
