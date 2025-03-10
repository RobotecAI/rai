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

import json
import logging
from typing import Annotated, Dict, List, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from rai.agents.spatiotemporal.spatiotemporal_agent import (
    Header,
    Point,
    Pose,
    PoseStamped,
    Quaternion,
    SpatioTemporalAgent,
    SpatioTemporalConfig,
)
from rai.communication import ROS2ARIConnector
from rai.messages.multimodal import HumanMultimodalMessage
from rai.tools.ros2.topics import GetROS2ImageTool


class ROS2SpatioTemporalConfig(SpatioTemporalConfig):
    camera_topics: List[str]
    robot_frame: str
    world_frame: str


class ROS2SpatioTemporalAgent(SpatioTemporalAgent):
    def __init__(self, config: ROS2SpatioTemporalConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.connector = ROS2ARIConnector()
        self.get_image_tool = GetROS2ImageTool(connector=self.connector)
        self.logger = logging.getLogger(__name__)

    def _get_images(self) -> Dict[Annotated[str, "camera topic"], str]:
        images: Dict[Annotated[str, "camera topic"], str] = {}
        for camera_topic in self.config.camera_topics:
            try:
                _, artifact = self.get_image_tool._run(topic=camera_topic)
                image = artifact["images"][0]
                images[camera_topic] = image
            except Exception as e:
                self.logger.warning(f"Error getting image from {camera_topic}: {e}")
        return images

    def _get_tf(self) -> Optional[PoseStamped]:
        try:
            tf_stamped = self.connector.get_transform(
                self.config.robot_frame, self.config.world_frame
            )
        except Exception as e:
            self.logger.warning(
                f"Error getting tf from {self.config.robot_frame} to {self.config.world_frame}: {e}"
            )
            return None
        ps = PoseStamped(
            header=Header(
                stamp=tf_stamped.header.stamp.sec
                + tf_stamped.header.stamp.nanosec * 1e-9,
                frame_id=tf_stamped.header.frame_id,
            ),
            pose=Pose(
                position=Point(
                    x=tf_stamped.transform.translation.x,
                    y=tf_stamped.transform.translation.y,
                    z=tf_stamped.transform.translation.z,
                ),
                orientation=Quaternion(
                    x=tf_stamped.transform.rotation.x,
                    y=tf_stamped.transform.rotation.y,
                    z=tf_stamped.transform.rotation.z,
                    w=tf_stamped.transform.rotation.w,
                ),
            ),
        )
        return ps

    def _get_image_text_descriptions(
        self, images: Dict[Annotated[str, "source"], str]
    ) -> Dict[Annotated[str, "source"], str]:
        text_description_prompt = SystemMessage(
            content="You are a helpful assistant that describes images."
        )
        text_descriptions: Dict[Annotated[str, "source"], str] = {}
        for source, image in images.items():
            human_message = HumanMultimodalMessage(
                content="Describe the image in detail.", images=[image]
            )
            ai_msg = cast(
                AIMessage,
                self.config.image_to_text_model.invoke(
                    [text_description_prompt, human_message]
                ),
            )
            if not isinstance(ai_msg.content, str):
                raise ValueError("AI message content is not a string")
            text_descriptions[source] = ai_msg.content

        return text_descriptions

    def _get_robots_history(self) -> str:
        # TODO: Implement this
        history: List[BaseMessage] = []
        return self._compress_context(history)

    def _compress_context(self, history: List[BaseMessage]) -> str:
        system_prompt = SystemMessage(
            content="You are a helpful assistant that compresses context. Your task is to compress the history of messages into a single message."
        )

        robots_history: List[Dict[str, str]] = []
        for msg in history:
            if not isinstance(msg.content, str):
                raise NotImplementedError("Only string content is supported")
            robots_history.append({"role": msg.type, "content": msg.content})
        if len(robots_history) == 0:
            return ""
        human_message = HumanMessage(content=json.dumps(robots_history))
        ai_msg = cast(
            AIMessage,
            self.config.context_compression_model.invoke(
                [system_prompt, human_message]
            ),
        )
        if not isinstance(ai_msg.content, str):
            raise ValueError("AI message content is not a string")
        return ai_msg.content
