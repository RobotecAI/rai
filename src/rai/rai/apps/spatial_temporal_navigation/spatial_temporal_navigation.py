# Copyright (C) 2024 Robotec.AI
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


import logging
import threading
import time
from typing import Dict, List
from uuid import uuid4

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from langchain_core.vectorstores.base import VectorStore
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from pymongo.collection import Collection
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

from rai.messages.multimodal import HumanMultimodalMessage
from rai.messages.utils import preprocess_image
from rai.tools.ros.tools import TF2TransformFetcher

logger = logging.getLogger(__name__)


class Pose(BaseModel):
    x: float = Field(..., description="The x coordinate of the position")
    y: float = Field(..., description="The y coordinate of the position")
    z: float = Field(..., description="The z coordinate of the position")


class Orientation(BaseModel):
    x: float = Field(..., description="The x coordinate of the orientation")
    y: float = Field(..., description="The y coordinate of the orientation")
    z: float = Field(..., description="The z coordinate of the orientation")
    w: float = Field(..., description="The w coordinate of the orientation")


class PositionStamped(BaseModel):
    timestamp: float
    position: Pose
    orientation: Orientation


class ImageStamped(BaseModel):
    timestamp: float
    image: str = Field(..., description="Base64 encoded image", repr=False)


class Scene(BaseModel):
    uuid: str


class Description(BaseModel):
    description: str
    objects: List[str]
    anomalies: List[str]


class Observation(BaseModel):
    uuid: str
    scene: Scene
    position_stamped: PositionStamped
    image_stamped: ImageStamped
    description: Description
    timestamp: float = Field(default_factory=time.time)


class VectorDatabaseEntry(BaseModel):
    text: str
    metadata: Dict[str, str]


def ros2_transform_stamped_to_position(
    transform_stamped: TransformStamped,
) -> PositionStamped:
    return PositionStamped(
        timestamp=transform_stamped.header.stamp.sec
        + transform_stamped.header.stamp.nanosec / 1e9,
        position=Pose(
            x=transform_stamped.transform.translation.x,
            y=transform_stamped.transform.translation.y,
            z=transform_stamped.transform.translation.z,
        ),
        orientation=Orientation(
            x=transform_stamped.transform.rotation.x,
            y=transform_stamped.transform.rotation.y,
            z=transform_stamped.transform.rotation.z,
            w=transform_stamped.transform.rotation.w,
        ),
    )


def ros2_image_to_image(ros2_image: Image) -> ImageStamped:
    logger.info("Converting ROS2 image to base64 image")
    bridge = CvBridge()
    cv2_image = bridge.imgmsg_to_cv2(ros2_image)
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    image_data = preprocess_image(cv2_image)
    return ImageStamped(
        timestamp=ros2_image.header.stamp.sec + ros2_image.header.stamp.nanosec / 1e9,
        image=image_data,
    )


def generate_description(image: ImageStamped) -> Description:
    logger.info("Generating LLM description")
    prompt = [
        HumanMultimodalMessage(
            content="Describe the image",
            images=[image.image],
        )
    ]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    description = llm.with_structured_output(Description).invoke(prompt)
    if not isinstance(description, Description):
        raise ValueError("Description is not a valid Description")
    return description


def build_observation(
    scene_uuid: str, position: TransformStamped, image: Image
) -> Observation:
    logger.info("Building observation")
    image_stamped: ImageStamped = ros2_image_to_image(image)
    position_stamped: PositionStamped = ros2_transform_stamped_to_position(position)
    description: Description = generate_description(image_stamped)
    return Observation(
        uuid=str(uuid4()),
        scene=Scene(uuid=scene_uuid),
        image_stamped=image_stamped,
        position_stamped=position_stamped,
        description=description,
    )


def observation_to_vector_database_entry(observation: Observation):
    return VectorDatabaseEntry(
        text=str(observation.description),
        metadata={"uuid": observation.uuid},
    )


def pipeline(
    vectorstore: VectorStore,
    observations_collection: Collection,
    image: Image,
    transform: TransformStamped,
):
    logger.info("Running pipeline")
    observation = build_observation(str(uuid4()), transform, image)
    vector_database_entry = observation_to_vector_database_entry(observation)

    logger.info("Adding to vectorstore")
    vectorstore.add_texts(
        texts=[vector_database_entry.text],
        metadatas=[vector_database_entry.metadata],
    )

    logger.info("Adding to MongoDB")
    observations_collection.insert_one(observation.model_dump())


class TransformGrabber:
    def __init__(self, target_frame: str, source_frame: str):
        self.transform_fetcher = TF2TransformFetcher(
            target_frame=target_frame, source_frame=source_frame
        )
        self.transform = None

    def run(self):
        while True:
            self.transform = self.transform_fetcher.get_data()


class ImageGrabber(Node):
    def __init__(self, image_topic: str):
        super().__init__("image_grabber")
        self.subscription = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )
        self.image: Image | None = None

    def image_callback(self, msg: Image):
        self.image = msg


def run(
    image_topic: str,
    source_frame: str,
    target_frame: str,
    vectorstore: VectorStore,
    observations_collection: Collection,
    time_between_observations: float = 5.0,
) -> None:
    transform_fetcher = TransformGrabber(
        target_frame=target_frame, source_frame=source_frame
    )
    image_grabber = ImageGrabber(image_topic)
    executor = SingleThreadedExecutor()
    executor.add_node(image_grabber)
    threading.Thread(target=transform_fetcher.run).start()
    threading.Thread(target=executor.spin).start()

    while True:
        image = image_grabber.image
        transform = transform_fetcher.transform
        if image is None or transform is None:
            time.sleep(0.1)
            continue
        threading.Thread(
            target=pipeline,
            args=(vectorstore, observations_collection, image, transform),
        ).start()
        time.sleep(time_between_observations)
