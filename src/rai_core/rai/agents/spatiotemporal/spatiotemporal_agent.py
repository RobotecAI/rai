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
import threading
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

IMAGE_DESCRIPTION_PROMPT = """
You are an expert at describing images.
When given an image, make sure to describe the objects in the image in detail.
"""

CONTEXT_COMPRESSION_PROMPT = """
You are an expert at compressing information.
When given a number of messages, compress them into a single message
retaining the information regarding the temporal context of the messages.
"""


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


class SpatioTemporalRecord(BaseModel):
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
        image_description_prompt: str = IMAGE_DESCRIPTION_PROMPT,
        context_compression_prompt: str = CONTEXT_COMPRESSION_PROMPT,
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
        self.image_description_prompt = image_description_prompt
        self.context_compression_prompt = context_compression_prompt
        self.db = MongoClient(self.config.db_url)[self.config.db_name]  # type: ignore
        self.collection = self.db[self.config.collection_name]  # type: ignore
        self.logger = logging.getLogger(__name__)
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @abstractmethod
    def _get_images(
        self,
    ) -> Dict[Annotated[str, "image source"], Annotated[str, "base64 encoded image"]]:
        """
        Abstract method to get images from sources (e.g. robot's cameras).

        Returns
        -------
        Dict[Annotated[str, "image source"], Annotated[str, "base64 encoded image"]]
            A dictionary mapping image sources to image data.
        """
        pass

    @abstractmethod
    def _get_tf(self) -> Optional[PoseStamped]:
        """
        Abstract method to get the transformation data (e.g. robot's pose).

        Returns
        -------
        Optional[PoseStamped]
            The transformation data, if available.
        """
        pass

    @abstractmethod
    def _get_robots_history(self) -> List[BaseMessage]:
        """
        Retrieve the robot's message history to provide
        a temporal context to the SpatioTemporalRecord samples.

        Returns
        -------
        List[BaseMessage]
            The history of messages.
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
        text_description_prompt = SystemMessage(content=self.image_description_prompt)
        text_descriptions: Dict[Annotated[str, "source"], str] = {}
        for source, image in images.items():
            human_message = HumanMultimodalMessage(
                content="Describe the image.", images=[image]
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
        system_prompt = SystemMessage(content=self.context_compression_prompt)

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

    def _insert_into_db(self, data: SpatioTemporalRecord):
        """
        Insert spatiotemporal data into the database.

        Parameters
        ----------
        data : SpatioTemporalRecord
            The spatiotemporal data to be inserted.
        """
        self.logger.info(
            f"Inserting data into database: {self.config.db_name}[{self.config.collection_name}]"
        )
        self.collection.insert_one(data.model_dump())  # type: ignore

    def _insert_into_vectorstore(self, data: SpatioTemporalRecord):
        """
        Insert embeddings of the spatiotemporal data into the vector store.

        Parameters
        ----------
        data : SpatioTemporalRecord
            The spatiotemporal data to be inserted.
        """
        self.logger.info("Inserting embeddings into vector store")

        self.config.vector_db.add_texts(  # type: ignore
            texts=[data.temporal_context + data.image_text_descriptions],
            metadatas=[{"id": data.id}],
            ids=[data.id],
        )

    def run(self):
        """
        Run the agent in a loop, executing tasks at specified intervals.
        """
        if self.thread is not None:
            raise RuntimeError("Agent is already running")

        def run_loop():
            while not self._stop_event.is_set():
                ts = time.time()
                self.logger.info("Starting new interval")
                self.on_interval()
                te = time.time()
                if te - ts > self.config.time_interval:
                    self.logger.warning(
                        f"Time interval exceeded. Expected {self.config.time_interval:.2f}s, got {te - ts:.2f}s"
                    )
                time.sleep(max(0, self.config.time_interval - (te - ts)))

            self._save_vector_store()

        thread = threading.Thread(target=run_loop)
        self.thread = thread
        self._stop_event.clear()
        thread.start()

    def stop(self):
        """Stop the agent's execution loop."""
        if self.thread is not None:
            self.logger.info("Stopping the agent. Please wait...")
            self._stop_event.set()
            self.thread.join()
            self.thread = None
            self._stop_event.clear()
            self.logger.info("Agent stopped")

    def on_interval(self):
        """
        Perform tasks at each interval, including data collection and insertion.
        """
        self.logger.info("Retrieving images")
        images = self._get_images()
        self.logger.info("Retrieving pose")
        tf = self._get_tf()
        temporal_context = self._get_robots_history()

        if tf is None and len(images) == 0:
            self.logger.warning(
                "No images or tf data to insert. Skipping this interval."
            )
            return
        self.logger.info("Generating image text descriptions")
        image_text_descriptions = self._get_image_text_descriptions(images)
        self.logger.info("Compressing temporal context")
        temporal_context = self._compress_context(temporal_context)

        record = SpatioTemporalRecord(
            timestamp=time.time(),
            images=images,
            tf=tf,
            temporal_context=temporal_context,
            image_text_descriptions=json.dumps(image_text_descriptions),
        )
        self._insert_into_db(record)
        self._insert_into_vectorstore(record)

    def _save_vector_store(self):
        """
        Save the vector store to disk.
        """
        from langchain_community.vectorstores import FAISS

        from rai.utils.model_initialization import load_config

        self.logger.info("Saving vector store")

        config = load_config()
        cast(FAISS, self.config.vector_db).save_local(config.vectorstore.uri)
