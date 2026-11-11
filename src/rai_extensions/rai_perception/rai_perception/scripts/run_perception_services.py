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

import os
import sys
import threading

from rai.agents import SyncAgent, run
from rai.communication.ros2 import ROS2Connector
from rai.communication.ros2.runtime import ROS2Runtime

from rai_perception.services import DetectionService, SegmentationService


class _StopOnShutdown(SyncAgent):
    def __init__(self, service: DetectionService | SegmentationService):
        super().__init__(name=type(service).__name__)
        self.service = service

    def teardown(self):
        self.service.stop()


def _connector(node_name: str) -> ROS2Connector:
    """A connector on its own node; --ros-args parameters apply to it."""
    connector = ROS2Connector(runtime=ROS2Runtime(node_name, cli_args=sys.argv))
    # ENABLE_LEGACY_SERVICE_NAMES=false (e.g. from a launch file) drops the
    # legacy service names; a --ros-args -p override still wins.
    enable_legacy = os.getenv("ENABLE_LEGACY_SERVICE_NAMES", "true").lower() == "true"
    connector.declare_parameter("enable_legacy_service_names", enable_legacy)
    return connector


def main():
    connectors = [_connector("detection_service"), _connector("segmentation_service")]
    classes = [DetectionService, SegmentationService]
    services: list = [None, None]

    def init(i: int) -> None:
        services[i] = classes[i](ros2_connector=connectors[i])

    # Initialize and download weights concurrently.
    threads = [threading.Thread(target=init, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    services = [s for s in services if s is not None]
    for service in services:
        service.run()
    run(*(_StopOnShutdown(s) for s in services))


if __name__ == "__main__":
    main()
