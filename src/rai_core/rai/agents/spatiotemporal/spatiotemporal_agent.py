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
import uuid
from abc import abstractmethod
from typing import Annotated, Any, Dict, List, Optional, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, ConfigDict, Field
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
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: Annotated[float, "timestamp"]
    images: Dict[Annotated[str, "camera topic"], str] = Field(repr=False)
    tf: Optional[PoseStamped]
    temporal_context: Annotated[str, "compressed history of messages"]
    image_text_descriptions: Annotated[str, "text descriptions of images"]


class SpatioTemporalConfig(BaseModel):
    db_url: str
    db_name: str
    collection_name: str
    image_to_text_model: BaseChatModel
    context_compression_model: BaseChatModel
    vector_db: VectorStore
    time_interval: float

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SpatioTemporalAgent(BaseAgent):
    def __init__(
        self,
        config: SpatioTemporalConfig,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ):
        """
        Initialize the SpatioTemporalAgent.

        Parameters
        ----------
        config : SpatioTemporalConfig
            Configuration for the spatiotemporal agent.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)  # type: ignore
        self.config = config

        self.db = MongoClient(self.config.db_url)[self.config.db_name]  # type: ignore
        self.collection = self.db[self.config.collection_name]  # type: ignore
        self.logger = logging.getLogger(__name__)

    def _insert_into_db(self, data: SpatioTemporalData):
        """
        Insert spatiotemporal data into the database.

        Parameters
        ----------
        data : SpatioTemporalData
            The spatiotemporal data to be inserted.
        """
        self.logger.info(
            f"Inserting data into database: {self.config.db_name}[{self.config.collection_name}]"
        )
        self.collection.insert_one(data.model_dump())  # type: ignore

    def _insert_into_vectorstore(self, data: SpatioTemporalData):
        """
        Insert embeddings of the spatiotemporal data into the vector store.

        Parameters
        ----------
        data : SpatioTemporalData
            The spatiotemporal data to be inserted.
        """
        self.logger.info("Inserting embeddings into vector store")

        print(
            self.config.vector_db.add_texts(  # type: ignore
                texts=[data.temporal_context + data.image_text_descriptions],
                metadatas=[{"id": data.id}],
                ids=[data.id],
            )
        )

    def run(self):
        """
        Run the agent in a loop, executing tasks at specified intervals.
        """
        while True:
            try:
                ts = time.time()
                self.logger.info("Starting new interval")
                self.on_interval()
                te = time.time()
                if te - ts > self.config.time_interval:
                    self.logger.warning(
                        f"Time interval exceeded. Expected {self.config.time_interval:.2f}s, got {te - ts:.2f}s"
                    )
                time.sleep(max(0, self.config.time_interval - (te - ts)))
            except KeyboardInterrupt:
                # seriously hacky
                from langchain_community.vectorstores import FAISS

                from rai.utils.model_initialization import load_config

                self.logger.info("Saving vector store")

                config = load_config()
                cast(FAISS, self.config.vector_db).save_local(config.vectorstore.uri)
                raise

    def on_interval(self):
        """
        Perform tasks at each interval, including data collection and insertion.
        """
        self.logger.info("Retrieving images")
        images = self._get_images()
        self.logger.info("Retrieving pose")
        tf = self._get_tf()
        if tf is None and len(images) == 0:
            self.logger.warning(
                "No images or tf data to insert. Skipping this interval."
            )
            return
        self.logger.info("Generating image text descriptions")
        image_text_descriptions = self._get_image_text_descriptions(images)
        self.logger.info("Retrieving temporal context")
        temporal_context = self._get_robots_history()

        data = SpatioTemporalData(
            timestamp=time.time(),
            images=images,
            tf=tf,
            temporal_context=temporal_context,
            image_text_descriptions=json.dumps(image_text_descriptions),
        )
        self._insert_into_db(data)
        self._insert_into_vectorstore(data)

    @abstractmethod
    def _get_images(
        self,
    ) -> Dict[Annotated[str, "image source"], Annotated[str, "base64 encoded image"]]:
        """
        Abstract method to get images from sources.

        Returns
        -------
        Dict[Annotated[str, "image source"], Annotated[str, "base64 encoded image"]]
            A dictionary mapping image sources to image data.
        """
        pass

    @abstractmethod
    def _get_tf(self) -> Optional[PoseStamped]:
        """
        Abstract method to get the transformation data.

        Returns
        -------
        Optional[PoseStamped]
            The transformation data, if available.
        """
        pass

    def _get_image_text_descriptions(
        self,
        images: Dict[Annotated[str, "source"], Annotated[str, "base64 encoded image"]],
    ) -> Dict[Annotated[str, "source"], Annotated[str, "text description"]]:
        """
        Generate text descriptions for images using a language model.

        Parameters
        ----------
        images : Dict[Annotated[str, "source"], Annotated[str, "base64 encoded image"]]
            A dictionary mapping image sources to image data.

        Returns
        -------
        Dict[Annotated[str, "source"], str]
            A dictionary mapping image sources to their text descriptions.
        """
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
            if not isinstance(ai_msg.content, str):  # type: ignore
                raise ValueError("AI message content is not a string")
            text_descriptions[source] = ai_msg.content

        return text_descriptions

    @abstractmethod
    def _get_robots_history(self) -> str:
        """
        Retrieve and compress the robot's message history.

        Returns
        -------
        str
            The compressed history of messages.
        """
        pass

    def _compress_context(self, history: List[BaseMessage]) -> str:
        """
        Compress a list of messages into a single context string.

        Parameters
        ----------
        history : List[BaseMessage]
            A list of messages to be compressed.

        Returns
        -------
        str
            The compressed context string.
        """
        system_prompt = SystemMessage(
            content="You are a helpful assistant that compresses context. Your task is to compress the history of messages into a single message."
        )

        robots_history: List[Dict[str, str]] = [
            {"role": msg.type, "content": msg.content}  # type: ignore
            for msg in history
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
        if not isinstance(ai_msg.content, str):  # type: ignore
            raise ValueError("AI message content is not a string")
        return ai_msg.content
