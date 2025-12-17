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
from typing import TYPE_CHECKING, Any, Callable, List, Tuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rai.communication.ros2.connectors.base import ROS2BaseConnector


def get_missing_entities(
    callable_get_entities: Callable[[], List[Tuple[str, Any]]],
    requested_entities: List[str],
) -> set[str]:
    available_entities = callable_get_entities()
    available_entity_names = [entity[0] for entity in available_entities]
    return set(requested_entities) - set(available_entity_names)


def wait_for_ros2_services(
    connector: "ROS2BaseConnector[Any]",
    requested_services: List[str],
    time_interval: float = 1.0,
    timeout: float = -1.0,
) -> None:
    """
    Wait until all requested ROS2 services are available.

    Args:
        connector: Connector providing service information.
        requested_services: List of service names to wait for.
        time_interval: Time interval (in seconds) to check for the services.
        timeout: Timeout in seconds. If set to -1 (default), there is no timeout.

    Raises:
        TimeoutError: If the services are not available within the timeout.
    """
    requested_services = [
        f"/{service}" if not service.startswith("/") else service
        for service in requested_services
    ]

    start_time = time.time()
    get_services = connector.get_services_names_and_types

    if timeout == -1:
        while True:
            missing = get_missing_entities(get_services, requested_services)
            if not missing:
                break
            logger.info(f"Waiting for services {missing}...")
            time.sleep(time_interval)
    else:
        while time.time() - start_time < timeout:
            missing = get_missing_entities(get_services, requested_services)
            if not missing:
                break
            logger.info(f"Waiting for services {missing}...")
            time.sleep(time_interval)
        else:
            available_services = get_services()
            service_names = [service[0] for service in available_services]
            missing = set(requested_services) - set(service_names)
            raise TimeoutError(
                f"Services {missing} not available within {timeout} seconds"
            )


def wait_for_ros2_topics(
    connector: "ROS2BaseConnector[Any]",
    requested_topics: List[str],
    time_interval: float = 1.0,
    timeout: float = -1.0,
) -> None:
    """
    Wait until all requested ROS2 topics are available.

    Args:
        connector: Connector providing topic information.
        requested_topics: List of topic names to wait for.
        time_interval: Time interval (in seconds) to check for the topics.
        timeout: Timeout in seconds. If set to -1 (default), there is no timeout.

    Raises:
        TimeoutError: If the topics are not available within the timeout.
    """
    requested_topics = [
        f"/{topic}" if not topic.startswith("/") else topic
        for topic in requested_topics
    ]

    start_time = time.time()
    get_topics = connector.get_topics_names_and_types

    if timeout == -1:
        while True:
            missing = get_missing_entities(get_topics, requested_topics)
            if not missing:
                break
            logger.info(f"Waiting for topics {missing}...")
            time.sleep(time_interval)
    else:
        while time.time() - start_time < timeout:
            missing = get_missing_entities(get_topics, requested_topics)
            if not missing:
                break
            logger.info(f"Waiting for topics {missing}...")
            time.sleep(time_interval)
        else:
            available_topics = get_topics()
            topic_names = [topic[0] for topic in available_topics]
            missing = set(requested_topics) - set(topic_names)
            raise TimeoutError(
                f"Topics {missing} not available within {timeout} seconds. Available topics: {topic_names}"
            )


def wait_for_ros2_actions(
    connector: "ROS2BaseConnector[Any]",
    requested_actions: List[str],
    time_interval: float = 1.0,
    timeout: float = -1.0,
) -> None:
    """
    Wait until all requested ROS2 actions are available.

    Args:
        connector: Connector providing action information.
        requested_actions: List of action names to wait for.
        time_interval: Time interval (in seconds) to check for the actions.
        timeout: Timeout in seconds. If set to -1 (default), there is no timeout.

    Raises:
        TimeoutError: If the actions are not available within the timeout.
    """
    requested_actions = [
        f"/{action}" if not action.startswith("/") else action
        for action in requested_actions
    ]

    start_time = time.time()
    get_actions = connector.get_actions_names_and_types

    if timeout == -1:
        while True:
            missing = get_missing_entities(get_actions, requested_actions)
            if not missing:
                break
            logger.info(f"Waiting for actions {missing}...")
            time.sleep(time_interval)
    else:
        while time.time() - start_time < timeout:
            missing = get_missing_entities(get_actions, requested_actions)
            if not missing:
                break
            logger.info(f"Waiting for actions {missing}...")
            time.sleep(time_interval)
        else:
            available_actions = get_actions()
            action_names = [action[0] for action in available_actions]
            missing = set(requested_actions) - set(action_names)
            raise TimeoutError(
                f"Actions {missing} not available within {timeout} seconds. Available actions: {action_names}"
            )
