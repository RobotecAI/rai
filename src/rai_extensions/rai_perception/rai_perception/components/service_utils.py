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

"""Calls to the detection and segmentation services, and graph checks."""

import time
from typing import Any, Iterable, Literal, Optional

from rai.communication.ros2 import ROS2Connector
from sensor_msgs.msg import Image

from rai_interfaces.srv import RAIGroundedSam, RAIGroundingDino

DETECTION_SERVICE = "/detection"
SEGMENTATION_SERVICE = "/segmentation"
# GroundingDINO and Grounded SAM can take long, especially on a CPU.
SERVICE_TIMEOUT_S = 60.0


async def detect(
    connector: ROS2Connector,
    image: Image,
    classes: str,
    *,
    service: str = DETECTION_SERVICE,
    box_threshold: float = 0.35,
    text_threshold: float = 0.45,
    timeout: float = SERVICE_TIMEOUT_S,
) -> Any:
    """Detections of classes ("cup, bottle") in image (a RAIGroundingDino response)."""
    request = RAIGroundingDino.Request(
        source_img=image,
        classes=classes,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    return await connector.call(service, request, timeout=timeout)


async def segment(
    connector: ROS2Connector,
    image: Image,
    detections: Any,
    *,
    service: str = SEGMENTATION_SERVICE,
    timeout: float = SERVICE_TIMEOUT_S,
) -> Any:
    """Masks of the detections in image (a RAIGroundedSam response)."""
    request = RAIGroundedSam.Request(source_img=image, detections=detections)
    return await connector.call(service, request, timeout=timeout)


Kind = Literal["service", "topic"]


def missing(
    connector: ROS2Connector, names: Iterable[str], kind: Kind = "service"
) -> list[str]:
    """The names not in the ROS graph."""
    if kind == "service":
        graph = connector.get_services_names_and_types()
    else:
        graph = connector.get_topics_names_and_types()
    available = {name for name, _ in graph}
    return [n for n in names if connector.resolve(n, kind) not in available]


def service_available(connector: ROS2Connector, service: str) -> bool:
    return not missing(connector, [service])


def wait_for(
    connector: ROS2Connector,
    names: Iterable[str],
    kind: Kind = "service",
    timeout: Optional[float] = None,
) -> list[str]:
    """Wait until every name is in the ROS graph; the names still missing
    after timeout (None waits forever)."""
    names = list(names)
    deadline = None if timeout is None else time.monotonic() + timeout
    while left := missing(connector, names, kind):
        if deadline is not None and time.monotonic() >= deadline:
            return left
        time.sleep(0.05)
    return []
