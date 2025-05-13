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
import time
from typing import TYPE_CHECKING, Any, List

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rai.communication.ros2.connectors.base import ROS2BaseConnector


def wait_for_ros2_services(
    connector: "ROS2BaseConnector[Any]",
    requested_services: List[str],
    time_interval: float = 1.0,
) -> None:
    # make sure the requested services start with '/'
    requested_services = [
        f"/{service}" if not service.startswith("/") else service
        for service in requested_services
    ]

    while True:
        available_services = connector.get_services_names_and_types()
        service_names = [service[0] for service in available_services]
        if set(requested_services).issubset(set(service_names)):
            break
        logger.info(
            f"Waiting for services {set(requested_services) - set(service_names)}..."
        )
        time.sleep(time_interval)


def wait_for_ros2_topics(
    connector: "ROS2BaseConnector[Any]",
    requested_topics: List[str],
    time_interval: float = 1.0,
) -> None:
    # make sure the requested topics start with '/'
    requested_topics = [
        f"/{topic}" if not topic.startswith("/") else topic
        for topic in requested_topics
    ]

    while True:
        available_topics = connector.get_topics_names_and_types()
        topic_names = [topic[0] for topic in available_topics]
        if set(requested_topics).issubset(set(topic_names)):
            break
        logger.info(f"Waiting for topics {set(requested_topics) - set(topic_names)}...")
        time.sleep(time_interval)


def wait_for_ros2_actions(
    connector: "ROS2BaseConnector[Any]",
    requested_actions: List[str],
    time_interval: float = 1.0,
) -> None:
    # make sure the requested actions start with '/'
    requested_actions = [
        f"/{action}" if not action.startswith("/") else action
        for action in requested_actions
    ]

    while True:
        available_actions = connector.get_actions_names_and_types()
        action_names = [action[0] for action in available_actions]
        if set(requested_actions).issubset(set(action_names)):
            break
        logger.info(
            f"Waiting for actions {set(requested_actions) - set(action_names)}..."
        )
        time.sleep(time_interval)
