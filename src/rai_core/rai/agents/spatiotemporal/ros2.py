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

import logging
from typing import Annotated, Dict, List, Optional

from langchain_core.messages import BaseMessage

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

    def _get_robots_history(self) -> str:
        # TODO: Implement this
        history: List[BaseMessage] = []
        return self._compress_context(history)
