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

from typing import Optional, Type

import rclpy
import sensor_msgs.msg
from langchain_core.pydantic_v1 import Field
from rclpy import Future

from rai.tools.ros import Ros2BaseInput, Ros2BaseTool
from rai_interfaces.msg import RAIDetectionArray
from rai_interfaces.srv import RAIGroundingDino


# --------------------- Inputs ---------------------
class Ros2GetDetectionInput(Ros2BaseInput):
    camera_topic: str = Field(
        ...,
        description="Ros2 topic for the camera image containing image to run detection on.",
    )
    object_name: list[str] = Field(
        ..., description="Natural language name of the objects to detect"
    )


class GetDetectionTool(Ros2BaseTool):
    name: str = "GetDetectionTool"
    description: str = (
        "A tool for detecting to a specified object using a ros2 action. The tool call might take some time to execute and is blocking - you will not be able to check their feedback, only will be informed about the result."
    )

    args_schema: Type[Ros2GetDetectionInput] = Ros2GetDetectionInput

    def _spin(self, future: Future) -> Optional[RAIDetectionArray]:
        rclpy.spin_once(self.node)
        if future.done():
            try:
                response = future.result()
            except Exception as e:
                self.node.get_logger().info("Service call failed %r" % (e,))
                raise Exception("Service call failed %r" % (e,))
            else:
                assert response is not None
                self.node.get_logger().info(f"{response.detections}")
                return response
        return None

    def _run(
        self,
        camera_topic: str,
        object_names: list[str],
    ):
        assert hasattr(self.node, "get_raw_message_from_topic")
        msg = self.node.get_raw_message_from_topic(camera_topic)
        camera_img_message = None
        if type(msg) is sensor_msgs.msg.Image:
            camera_img_message = msg
        else:
            raise Exception("Received wrong message")

        cli = self.node.create_client(RAIGroundingDino, "grounding_dino_classify")
        while not cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("service not available, waiting again...")
        req = RAIGroundingDino.Request()
        req.source_img = camera_img_message
        req.classes = " , ".join(object_names)
        req.box_threshold = 0.4
        req.text_threshold = 0.4

        future = cli.call_async(req)

        while rclpy.ok():
            resolved = self._spin(future)
            if resolved is not None:
                return f"I have detected the following items in the picture {resolved.detections.detection_classes}"

        return "Failed to get detection"
