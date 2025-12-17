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
from typing import TYPE_CHECKING, Any, Callable, List

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rai.communication.ros2.connectors.base import ROS2BaseConnector


def ensure_slash(name: str) -> str:
    return name if name.startswith("/") else f"/{name}"


def get_missing_entities(
    callable_get_entities: Callable[[], List[str]],
    requested_entities: List[str],
) -> set[str]:
    requested_set = set(requested_entities)
    available_set = set(callable_get_entities())
    return requested_set - available_set


def wait_for_ros2_entities(
    requested: List[str],
    get_entities: Callable[[], List[str]],
    entity_type: str = "entity",
    time_interval: float = 1.0,
    timeout: float = 0.0,
) -> None:
    requested = [ensure_slash(name) for name in requested]

    if timeout < 0:
        raise ValueError("Timeout must be 0 (wait forever) or a positive value.")
    start_time = time.time()

    while timeout == 0 or time.time() - start_time < timeout:
        missing = get_missing_entities(get_entities, requested)
        if not missing:
            return
        logger.info(f"Waiting for {entity_type} {missing}...")
        time.sleep(time_interval)

    available_entities = get_entities()

    missing = set(requested) - set(available_entities)
    raise TimeoutError(
        f"{entity_type.capitalize()} {missing} not available within {timeout} seconds. "
    )


def wait_for_ros2_services(
    connector: "ROS2BaseConnector[Any]",
    requested_services: List[str],
    time_interval: float = 1.0,
    timeout: float = 0.0,
) -> None:
    """Wait for specified ROS2 services to become available.

    Parameters
    ----------
    connector : ROS2BaseConnector[Any]
        ROS2 connector providing access to system service information.
    requested_services : List[str]
        List of service names to wait for.
    time_interval : float, optional
        Wait interval (seconds) between polling checks, by default 1.0.
    timeout : float, optional
        Timeout in seconds. If set to 0 (default), wait indefinitely.

    Raises
    ------
    TimeoutError
        If not all requested services become available within the timeout period.
    ValueError
        If `timeout` is set to a negative value.

    Notes
    -----
    - If `timeout == 0`, this function waits indefinitely and never raises `TimeoutError`.
    - If `timeout > 0`, a `TimeoutError` is raised if the services are not found in time.
    - Raises ValueError if timeout < 0.
    """
    get_services = lambda: [
        service[0] for service in connector.get_services_names_and_types()
    ]
    return wait_for_ros2_entities(
        requested_services,
        get_services,
        entity_type="service",
        time_interval=time_interval,
        timeout=timeout,
    )


def wait_for_ros2_topics(
    connector: "ROS2BaseConnector[Any]",
    requested_topics: List[str],
    time_interval: float = 1.0,
    timeout: float = 0.0,
) -> None:
    """Wait for specified ROS2 topics to become available.

    Parameters
    ----------
    connector : ROS2BaseConnector[Any]
        ROS2 connector providing access to system topic information.
    requested_topics : List[str]
        List of topic names to wait for.
    time_interval : float, optional
        Wait interval (seconds) between polling checks, by default 1.0.
    timeout : float, optional
        Timeout in seconds. If set to 0 (default), wait indefinitely.

    Raises
    ------
    TimeoutError
        If not all requested topics become available within the timeout period.
    ValueError
        If `timeout` is set to a negative value.

    Notes
    -----
    - If `timeout == 0`, this function waits indefinitely and never raises `TimeoutError`.
    - If `timeout > 0`, a `TimeoutError` is raised if the topics are not found in time.
    - Raises ValueError if timeout < 0.
    """
    get_topics = lambda: [topic[0] for topic in connector.get_topics_names_and_types()]
    return wait_for_ros2_entities(
        requested_topics,
        get_topics,
        entity_type="topic",
        time_interval=time_interval,
        timeout=timeout,
    )


def wait_for_ros2_actions(
    connector: "ROS2BaseConnector[Any]",
    requested_actions: List[str],
    time_interval: float = 1.0,
    timeout: float = 0.0,
) -> None:
    """Wait for specified ROS2 actions to become available.

    Parameters
    ----------
    connector : ROS2BaseConnector[Any]
        ROS2 connector providing access to system action information.
    requested_actions : List[str]
        List of action names to wait for.
    time_interval : float, optional
        Wait interval (seconds) between polling checks, by default 1.0.
    timeout : float, optional
        Timeout in seconds. If set to 0 (default), wait indefinitely.

    Raises
    ------
    TimeoutError
        If not all requested actions become available within the timeout period.
    ValueError
        If `timeout` is set to a negative value.

    Notes
    -----
    - If `timeout == 0`, this function waits indefinitely and never raises `TimeoutError`.
    - If `timeout > 0`, a `TimeoutError` is raised if the actions are not found in time.
    - Raises ValueError if timeout < 0.
    """
    get_actions = lambda: [
        action[0] for action in connector.get_actions_names_and_types()
    ]
    return wait_for_ros2_entities(
        requested_actions,
        get_actions,
        entity_type="action",
        time_interval=time_interval,
        timeout=timeout,
    )
