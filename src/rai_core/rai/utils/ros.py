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

import time
from typing import Callable, Dict, List, Optional, Tuple

import rclpy.callback_groups
import rclpy.node
from rclpy.action.graph import get_action_names_and_types


class ROS2Context:
    """A context manager and decorator for ROS 2 node lifecycle management.

    This class ensures proper initialization and cleanup of the ROS 2 system,
    whether used as a context manager or decorator.

    Parameters
    ----------
    domain_id : int, optional
        ROS domain ID for network segmentation. Default is None.

    Attributes
    ----------
    domain_id : int or None
        The ROS domain ID used for initialization
    _initialized : bool
        Internal flag tracking ROS 2 initialization state

    Methods
    -------
    is_initialized
        Property that returns the current ROS 2 initialization state

    Examples
    --------
    Using as a context manager:

    >>> with ROS2Context() as ctx:
    ...     node = rclpy.create_node('my_node')
    ...     rclpy.spin_once(node)

    Using as a decorator:

    >>> @ROS2Context()
    ... def run_ros_node():
    ...     node = rclpy.create_node('my_node')
    ...     rclpy.spin_once(node)
    """

    def __init__(self, *, domain_id: Optional[int] = None):
        """Initialize the ROS2Context.

        Parameters
        ----------
        domain_id : int, optional
            ROS domain ID for network segmentation
        """
        self.domain_id = domain_id
        self._initialized = False

    def __enter__(self):
        """Enter the context manager.

        Returns
        -------
        ROS2Context
            The initialized context manager instance

        Raises
        ------
        RuntimeError
            If ROS 2 initialization fails
        """
        try:
            rclpy.init(domain_id=self.domain_id)
            self._initialized = True
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ROS 2: {e}") from e

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager and cleanup ROS 2.

        Parameters
        ----------
        exc_type : type
            The type of the exception that occurred, if any
        exc_value : Exception
            The exception instance that occurred, if any
        traceback : traceback
            The traceback of the exception that occurred, if any
        """
        if self._initialized:
            try:
                rclpy.shutdown()
            except Exception as e:
                import logging

                logging.error(f"Error during ROS 2 shutdown: {e}")
        self._initialized = False

    def __call__(self, func):
        """Make the class callable as a decorator.

        Parameters
        ----------
        func : callable
            The function to be wrapped

        Returns
        -------
        callable
            The wrapped function with ROS 2 context management
        """

        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        # Preserve the original function's metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    @property
    def is_initialized(self) -> bool:
        """Check if ROS 2 is currently initialized.

        Returns
        -------
        bool
            True if ROS 2 is initialized, False otherwise
        """
        return self._initialized


class NodeDiscovery:
    def __init__(
        self,
        node: rclpy.node.Node,
        allowlist: Optional[List[str]] = None,
        period_sec: float = 0.5,
        setters: Optional[List[Callable]] = None,
    ) -> None:
        self.period_sec = period_sec
        self.node = node

        self.topics_and_types: Dict[str, str] = dict()
        self.services_and_types: Dict[str, str] = dict()
        self.actions_and_types: Dict[str, str] = dict()
        self.allowlist: Optional[List[str]] = allowlist

        self.timer = self.node.create_timer(
            self.period_sec,
            self.discovery_callback,
            callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
        )

        # callables (e.g. fun(x: NodeDiscovery)) that will receive the discovery
        # info on every timer callback. This allows to register other entities that
        # needs up-to-date discovery info
        if setters is None:
            self.setters = list()
        else:
            self.setters = setters

        # make first callback as fast as possible
        # sleep before first callback due ros discovery issue: https://github.com/ros2/ros2/issues/1057
        time.sleep(0.5)
        self.discovery_callback()

    def add_setter(self, setter: Callable):
        self.setters.append(setter)

    def discovery_callback(self):
        self.__set(
            self.node.get_topic_names_and_types(),
            self.node.get_service_names_and_types(),
            get_action_names_and_types(self.node),
        )
        for setter in self.setters:
            setter(self)

    def __set(self, topics, services, actions):
        def to_dict(info: List[Tuple[str, List[str]]]) -> Dict[str, str]:
            return {k: v[0] for k, v in info}

        self.topics_and_types = to_dict(topics)
        self.services_and_types = to_dict(services)
        self.actions_and_types = to_dict(actions)
        if self.allowlist is not None:
            self.__filter(self.allowlist)

    def __filter(self, allowlist: List[str]):
        for d in [
            self.topics_and_types,
            self.services_and_types,
            self.actions_and_types,
        ]:
            to_remove = [k for k in d if k not in allowlist]
            for k in to_remove:
                d.pop(k)

    def dict(self):
        return {
            "topics_and_types": self.topics_and_types,
            "services_and_types": self.services_and_types,
            "actions_and_types": self.actions_and_types,
        }
