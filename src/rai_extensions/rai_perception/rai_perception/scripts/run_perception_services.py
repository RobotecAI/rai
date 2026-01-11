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


import rclpy
from rai.agents import wait_for_shutdown
from rai.communication.ros2 import ROS2Connector

from rai_perception.services import DetectionService, SegmentationService


def main():
    rclpy.init()

    # Create ROS2 connectors for each service
    # TODO(juliaj): Re-evaluate executor_type choice (single_threaded vs multi_threaded)
    # Current: single_threaded for consistency with BaseVisionService pattern
    # Consider: multi_threaded if concurrent request handling is needed
    detection_connector = ROS2Connector(
        "detection_service", executor_type="single_threaded"
    )
    segmentation_connector = ROS2Connector(
        "segmentation_service", executor_type="single_threaded"
    )

    # Services read model_name from ROS2 params (defaults: "grounding_dino", "grounded_sam")
    detection_service = DetectionService(ros2_connector=detection_connector)
    segmentation_service = SegmentationService(ros2_connector=segmentation_connector)

    detection_service.run()
    segmentation_service.run()

    wait_for_shutdown([detection_service, segmentation_service])
    rclpy.shutdown()


if __name__ == "__main__":
    main()
