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

import threading
import time
import uuid
from functools import partial
from typing import Any, Callable, Dict, Final, List, Literal, Optional, Tuple, TypeVar

import rclpy
import rclpy.executors
import rclpy.node
import rclpy.time
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from tf2_ros import Buffer, LookupException, TransformListener, TransformStamped

from rai.communication import BaseConnector
from rai.communication.ros2.api import (
    ROS2ActionAPI,
    ROS2ServiceAPI,
    ROS2TopicAPI,
)
from rai.communication.ros2.connectors.action_mixin import ROS2ActionMixin
from rai.communication.ros2.connectors.service_mixin import ROS2ServiceMixin
from rai.communication.ros2.messages import ROS2Message

T = TypeVar("T", bound=ROS2Message)


class ROS2BaseConnector(ROS2ActionMixin, ROS2ServiceMixin, BaseConnector[T]):
    """ROS2-specific implementation of the BaseConnector.

    This connector provides functionality for ROS2 communication through topics,
    services, and actions, as well as TF (Transform) operations.

    Parameters
    ----------
    node_name : str, optional
        Name of the ROS2 node. If not provided, generates a unique name with UUID.
    destroy_subscribers : bool, optional
        Whether to destroy subscribers after receiving a message, by default False.
    executor_type : Literal["single_threaded", "multi_threaded"], optional
        Type of executor to use for processing ROS2 callbacks, by default "multi_threaded".

    Methods
    -------
    get_topics_names_and_types()
        Get list of available topics and their message types.
    get_services_names_and_types()
        Get list of available services and their types.
    get_actions_names_and_types()
        Get list of available actions and their types.
    send_message(message, target, msg_type, auto_qos_matching=True, qos_profile=None, **kwargs)
        Send a message to a specified topic.
    receive_message(source, timeout_sec=1.0, msg_type=None, auto_topic_type=True, **kwargs)
        Receive a message from a specified topic.
    wait_for_transform(tf_buffer, target_frame, source_frame, timeout_sec=1.0)
        Wait for a transform to become available.
    get_transform(target_frame, source_frame, timeout_sec=5.0)
        Get the transform between two frames.
    create_service(service_name, on_request, on_done=None, service_type, **kwargs)
        Create a ROS2 service.
    create_action(action_name, generate_feedback_callback, action_type, **kwargs)
        Create a ROS2 action server.
    shutdown()
        Clean up resources and shut down the connector.

    Notes
    -----
    Threading Model:
        The connector creates an executor that runs in a dedicated thread.
        This executor processes all ROS2 callbacks and operations asynchronously.

    Subscriber Lifecycle:
        The `destroy_subscribers` parameter controls subscriber cleanup behavior:
        - True: Subscribers are destroyed after receiving a message
            - Pros: Better resource utilization
            - Cons: Known stability issues (see: https://github.com/ros2/rclpy/issues/1142)
        - False (default): Subscribers remain active after message reception
            - Pros: More stable operation, avoids potential crashes
            - Cons: May lead to memory/performance overhead from inactive subscribers
    """

    def __init__(
        self,
        node_name: str = f"rai_ros2_connector_{str(uuid.uuid4())[-12:]}",
        destroy_subscribers: bool = False,
        executor_type: Literal["single_threaded", "multi_threaded"] = "multi_threaded",
    ):
        """Initialize the ROS2BaseConnector.

        Parameters
        ----------
        node_name : str, optional
            Name of the ROS2 node. If not provided, generates a unique name with UUID.
        destroy_subscribers : bool, optional
            Whether to destroy subscribers after receiving a message, by default False.
        executor_type : Literal["single_threaded", "multi_threaded"], optional
            Type of executor to use for processing ROS2 callbacks, by default "multi_threaded".

        Raises
        ------
        ValueError
            If an invalid executor type is provided.
        """
        super().__init__()

        if not rclpy.ok():
            rclpy.init()
            self.logger.warning(
                "Auto-initializing ROS2, but manual initialization is recommended. "
                "For better control and predictability, call rclpy.init() or ROS2Context before creating this connector."
            )
        self._executor_type = executor_type
        self._node = Node(node_name)
        self._topic_api = ROS2TopicAPI(self._node, destroy_subscribers)
        self._service_api = ROS2ServiceAPI(self._node)
        self._actions_api = ROS2ActionAPI(self._node)
        self._tf_buffer = Buffer(node=self._node)
        self._tf_listener = TransformListener(self._tf_buffer, self._node)

        self._executor_performance_time_delta = 1.0
        self._executor_performance_timer = self._node.create_timer(
            self._executor_performance_time_delta, self._executor_performance_callback
        )
        self._performance_warning_threshold_multiplier: Final[float] = 1.1
        self._available_executors: Final[set[str]] = {
            "MultiThreadedExecutor",
            "SingleThreadedExecutor",
        }
        if self._executor_type == "multi_threaded":
            self._executor = MultiThreadedExecutor()
        elif self._executor_type == "single_threaded":
            self._executor = SingleThreadedExecutor()
        else:
            raise ValueError(f"Invalid executor type: {self._executor_type}")

        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin)
        self._thread.start()
        self.last_executor_performance_time = time.time()

        # cache for last received messages
        self.last_msg: Dict[str, T] = {}

    def _executor_performance_callback(self) -> None:
        """Monitor executor performance and log warnings if it falls behind schedule.

        This callback checks if the executor is running slower than expected and logs
        a warning with suggestions for alternative executors if performance issues
        are detected.
        """
        current_time = time.time()
        time_behind = (
            current_time
            - self.last_executor_performance_time
            - self._executor_performance_time_delta
        )
        threshold = (
            self._executor_performance_time_delta
            * self._performance_warning_threshold_multiplier
        )

        if time_behind > threshold:
            alternative_executors = self._available_executors - {
                self._executor.__class__.__name__
            }

            self.logger.warning(
                f"{self._executor.__class__.__name__} is {time_behind:.2f} seconds behind. "
                f"If you see this message frequently, consider switching to {', '.join(alternative_executors)}."
            )
            self.last_executor_performance_time = current_time
        else:
            self.last_executor_performance_time = current_time

    def _last_message_callback(self, source: str, msg: T):
        """Store the last received message for a given source.

        Parameters
        ----------
        source : str
            The topic source identifier.
        msg : T
            The received message.
        """
        self.last_msg[source] = msg

    def get_topics_names_and_types(self) -> List[Tuple[str, List[str]]]:
        """Get list of available topics and their message types.

        Returns
        -------
        List[Tuple[str, List[str]]]
            List of tuples containing topic names and their corresponding message types.
        """
        return self._topic_api.get_topic_names_and_types()

    def get_services_names_and_types(self) -> List[Tuple[str, List[str]]]:
        """Get list of available services and their types.

        Returns
        -------
        List[Tuple[str, List[str]]]
            List of tuples containing service names and their corresponding types.
        """
        return self._service_api.get_service_names_and_types()

    def get_actions_names_and_types(self) -> List[Tuple[str, List[str]]]:
        """Get list of available actions and their types.

        Returns
        -------
        List[Tuple[str, List[str]]]
            List of tuples containing action names and their corresponding types.
        """
        return self._actions_api.get_action_names_and_types()

    def send_message(
        self,
        message: T,
        target: str,
        *,
        msg_type: str,
        auto_qos_matching: bool = True,
        qos_profile: Optional[QoSProfile] = None,
        **kwargs: Any,
    ):
        """Send a message to a specified topic.

        Parameters
        ----------
        message : T
            The message to send.
        target : str
            The target topic name.
        msg_type : str
            The ROS2 message type.
        auto_qos_matching : bool, optional
            Whether to automatically match QoS profiles, by default True.
        qos_profile : Optional[QoSProfile], optional
            Custom QoS profile to use, by default None.
        **kwargs : Any
            Additional keyword arguments.
        """
        self._topic_api.publish(
            topic=target,
            msg_content=message.payload,
            msg_type=msg_type,
            auto_qos_matching=auto_qos_matching,
            qos_profile=qos_profile,
        )

    def general_callback_preprocessor(self, message: Any) -> T:
        """Preprocess a raw ROS2 message into a connector message.

        Parameters
        ----------
        message : Any
            The raw ROS2 message.

        Returns
        -------
        T
            The preprocessed message.
        """
        return self.T_class(payload=message, metadata={"msg_type": str(type(message))})

    def register_callback(
        self,
        source: str,
        callback: Callable[[T | Any], None],
        raw: bool = False,
        *,
        msg_type: Optional[str] = None,
        qos_profile: Optional[QoSProfile] = None,
        auto_qos_matching: bool = True,
        **kwargs: Any,
    ) -> str:
        """Register a callback for a topic.

        Parameters
        ----------
        source : str
            The topic to subscribe to.
        callback : Callable[[T | Any], None]
            The callback function to execute when a message is received.
        raw : bool, optional
            Whether to pass raw messages to the callback, by default False.
        msg_type : Optional[str], optional
            The ROS2 message type, by default None.
        qos_profile : Optional[QoSProfile], optional
            Custom QoS profile to use, by default None.
        auto_qos_matching : bool, optional
            Whether to automatically match QoS profiles, by default True.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        str
            The callback ID.
        """
        exists = self._topic_api.subscriber_exists(source)
        if not exists:
            self._topic_api.create_subscriber(
                topic=source,
                msg_type=msg_type,
                callback=partial(self.general_callback, source),
                qos_profile=qos_profile,
                auto_qos_matching=auto_qos_matching,
            )
        return super().register_callback(source, callback, raw=raw)

    def receive_message(
        self,
        source: str,
        timeout_sec: float = 1.0,
        *,
        msg_type: Optional[str] = None,
        qos_profile: Optional[QoSProfile] = None,
        auto_qos_matching: bool = True,
        **kwargs: Any,
    ) -> T:
        """Receive a message from a topic.

        Parameters
        ----------
        source : str
            The topic to receive from.
        timeout_sec : float, optional
            Timeout in seconds, by default 1.0.
        msg_type : Optional[str], optional
            The ROS2 message type, by default None.
        qos_profile : Optional[QoSProfile], optional
            Custom QoS profile to use, by default None.
        auto_qos_matching : bool, optional
            Whether to automatically match QoS profiles, by default True.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        T
            The received message.

        Raises
        ------
        TimeoutError
            If no message is received within the timeout period.
        """
        if self._topic_api.subscriber_exists(source):
            # trying to hit cache first
            if source in self.last_msg:
                if self.last_msg[source].timestamp > time.time() - timeout_sec:
                    return self.last_msg[source]
        else:
            self._topic_api.create_subscriber(
                topic=source,
                callback=partial(self.general_callback, source),
                msg_type=msg_type,
                qos_profile=qos_profile,
                auto_qos_matching=auto_qos_matching,
            )
            self.register_callback(source, partial(self._last_message_callback, source))

        start_time = time.time()
        # wait for the message to be received
        while time.time() - start_time < timeout_sec:
            if source in self.last_msg:
                return self.last_msg[source]
            time.sleep(0.1)
        else:
            raise TimeoutError(
                f"Message from {source} not received in {timeout_sec} seconds"
            )

    @staticmethod
    def wait_for_transform(
        tf_buffer: Buffer,
        target_frame: str,
        source_frame: str,
        timeout_sec: float = 1.0,
    ) -> bool:
        """Wait for a transform to become available.

        Parameters
        ----------
        tf_buffer : Buffer
            The TF buffer to check.
        target_frame : str
            The target frame.
        source_frame : str
            The source frame.
        timeout_sec : float, optional
            Timeout in seconds, by default 1.0.

        Returns
        -------
        bool
            True if the transform is available, False otherwise.
        """
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            if tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time()):
                return True
            time.sleep(0.1)
        return False

    def get_transform(
        self,
        target_frame: str,
        source_frame: str,
        timeout_sec: float = 5.0,
    ) -> TransformStamped:
        """Get the transform between two frames.

        Parameters
        ----------
        target_frame : str
            The target frame.
        source_frame : str
            The source frame.
        timeout_sec : float, optional
            Timeout in seconds, by default 5.0.

        Returns
        -------
        TransformStamped
            The transform between the frames.

        Raises
        ------
        LookupException
            If the transform is not available within the timeout period.
        """
        transform_available = self.wait_for_transform(
            self._tf_buffer, target_frame, source_frame, timeout_sec
        )
        if not transform_available:
            raise LookupException(
                f"Could not find transform from {source_frame} to {target_frame} in {timeout_sec} seconds"
            )
        transform: TransformStamped = self._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=int(timeout_sec)),
        )

        return transform

    def create_service(
        self,
        service_name: str,
        on_request: Callable[[Any, Any], Any],
        on_done: Optional[Callable[[Any, Any], Any]] = None,
        *,
        service_type: str,
        **kwargs: Any,
    ) -> str:
        """Create a ROS2 service.

        Parameters
        ----------
        service_name : str
            The name of the service.
        on_request : Callable[[Any, Any], Any]
            Callback function to handle service requests.
        on_done : Optional[Callable[[Any, Any], Any]], optional
            Callback function called when service is terminated, by default None.
        service_type : str
            The ROS2 service type.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        str
            The service handle.
        """
        return self._service_api.create_service(
            service_name=service_name,
            callback=on_request,
            service_type=service_type,
            **kwargs,
        )

    def create_action(
        self,
        action_name: str,
        generate_feedback_callback: Callable,
        *,
        action_type: str,
        **kwargs: Any,
    ) -> str:
        """Create a ROS2 action server.

        Parameters
        ----------
        action_name : str
            The name of the action.
        generate_feedback_callback : Callable
            Callback function to generate feedback during action execution.
        action_type : str
            The ROS2 action type.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        str
            The action handle.
        """
        return self._actions_api.create_action_server(
            action_name=action_name,
            action_type=action_type,
            execute_callback=generate_feedback_callback,
            **kwargs,
        )

    @property
    def node(self) -> Node:
        """Get the ROS2 node.

        Returns
        -------
        Node
            The ROS2 node instance.
        """
        return self._node

    def shutdown(self):
        """Shutdown the connector and clean up resources.

        This method:
        1. Unregisters the TF listener
        2. Destroys the ROS2 node
        3. Shuts down the action API
        4. Shuts down the topic API
        5. Shuts down the executor
        6. Joins the executor thread
        """
        self._tf_listener.unregister()
        self._node.destroy_node()
        self._actions_api.shutdown()
        self._topic_api.shutdown()
        self._executor.shutdown()
        self._thread.join()
