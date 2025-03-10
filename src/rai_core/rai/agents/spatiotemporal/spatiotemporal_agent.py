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
import time
from abc import abstractmethod
from typing import Annotated, Dict, List, Optional, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from pymongo import MongoClient

from rai.agents.base import BaseAgent
from rai.messages.multimodal import HumanMultimodalMessage


class Header(BaseModel):
    stamp: Annotated[float, "timestamp"]
    frame_id: str


class Point(BaseModel):
    x: float
    y: float
    z: float


class Quaternion(BaseModel):
    x: float
    y: float
    z: float
    w: float


class Pose(BaseModel):
    position: Point
    orientation: Quaternion


class PoseStamped(BaseModel):
    header: Header
    pose: Pose


class SpatioTemporalData(BaseModel):
    timestamp: Annotated[float, "timestamp"]
    images: Dict[Annotated[str, "camera topic"], str]
    tf: Optional[PoseStamped]
    temporal_context: Annotated[str, "compressed history of messages"]
    image_text_descriptions: Annotated[str, "text descriptions of images"]


class SpatioTemporalConfig(BaseModel):
    db_url: str
    db_name: str
    collection_name: str
    image_to_text_model: BaseChatModel
    context_compression_model: BaseChatModel
    time_interval: float


class SpatioTemporalAgent(BaseAgent):
    def __init__(self, config: SpatioTemporalConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.db = MongoClient(self.config.db_url)[self.config.db_name]
        self.collection = self.db[self.config.collection_name]
        self.logger = logging.getLogger(__name__)

    def insert_into_db(self, data: SpatioTemporalData):
        self.logger.info(
            f"Inserting data into database: {self.config.db_name}[{self.config.collection_name}]"
        )
        self.collection.insert_one(data.model_dump())

    def run(self):
        while True:
            ts = time.time()
            self.logger.info("Running on interval")
            self.on_interval()
            te = time.time()
            if te - ts > self.config.time_interval:
                self.logger.warning(
                    f"Time interval exceeded. Expected {self.config.time_interval:.2f}s, got {te - ts:.2f}s"
                )
            time.sleep(max(0, self.config.time_interval - (te - ts)))

    def on_interval(self):
        images = self._get_images()
        tf = self._get_tf()
        if tf is None and len(images) == 0:
            self.logger.warning(
                "No images or tf data to insert. Skipping this interval."
            )
            return
        image_text_descriptions = self._get_image_text_descriptions(images)
        temporal_context = self._get_robots_history()
        data = SpatioTemporalData(
            timestamp=time.time(),
            images=images,
            tf=tf,
            temporal_context=temporal_context,
            image_text_descriptions=json.dumps(image_text_descriptions),
        )
        self.insert_into_db(data)

    @abstractmethod
    def _get_images(self) -> Dict[Annotated[str, "image source"], str]:
        pass

    @abstractmethod
    def _get_tf(self) -> Optional[PoseStamped]:
        pass

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

        robots_history: List[Dict[str, str]] = [
            {"role": msg.type, "content": msg.content} for msg in history
        ]
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
