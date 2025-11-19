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


from pathlib import Path

from rai_interfaces.msg import RAIDetectionArray
from rai_perception.agents.base_vision_agent import BaseVisionAgent
from rai_perception.vision_markup.boxer import GDBoxer

GDINO_NODE_NAME = "grounding_dino"
GDINO_SERVICE_NAME = "grounding_dino_classify"


class GroundingDinoAgent(BaseVisionAgent):
    WEIGHTS_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    WEIGHTS_FILENAME = "groundingdino_swint_ogc.pth"

    def __init__(
        self,
        weights_root_path: str | Path = Path.home() / Path(".cache/rai"),
        ros2_name: str = GDINO_NODE_NAME,
    ):
        super().__init__(weights_root_path, ros2_name)
        self._boxer = self._load_model_with_error_handling(GDBoxer)
        self.logger.info(f"{self.__class__.__name__} initialized")

    def run(self):
        self.ros2_connector.create_service(
            GDINO_SERVICE_NAME,
            self._classify_callback,
            service_type="rai_interfaces/srv/RAIGroundingDino",
        )

    def _classify_callback(self, request, response: RAIDetectionArray):
        self.logger.info(
            f"Request received: {request.classes}, {request.box_threshold}, {request.text_threshold}"
        )

        class_array = request.classes.split(",")
        class_array = [class_name.strip() for class_name in class_array]
        class_dict = {class_name: i for i, class_name in enumerate(class_array)}

        boxes = self._boxer.get_boxes(
            request.source_img,
            class_array,
            request.box_threshold,
            request.text_threshold,
        )

        ts = self.ros2_connector._node.get_clock().now().to_msg()
        response.detections.detections = [  # type: ignore
            box.to_detection_msg(class_dict, ts)
            for box in boxes  # type: ignore
        ]
        response.detections.header.stamp = ts  # type: ignore
        response.detections.detection_classes = class_array  # type: ignore

        return response
