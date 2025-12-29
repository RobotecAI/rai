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

import importlib
import logging
from typing import (
    Any,
    Dict,
    List,
    Protocol,
    Tuple,
    Type,
    runtime_checkable,
)

import rclpy
import rclpy.node
import rosidl_runtime_py.set_message
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    LivelinessPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.topic_endpoint_info import TopicEndpointInfo
from rosidl_parser.definition import NamespacedType
from rosidl_runtime_py.import_message import import_message_from_namespaced_type
from rosidl_runtime_py.utilities import (
    get_namespaced_type,
    is_action,
    is_message,
    is_service,
)

from rai.communication.ros2.api.conversion import import_message_from_str


@runtime_checkable
class IROS2Message(Protocol):
    __slots__: list

    def get_fields_and_field_types(self) -> dict: ...


class BaseROS2API:
    node: rclpy.node.Node

    @staticmethod
    def adapt_requests_to_offers(publisher_info: List[TopicEndpointInfo]) -> QoSProfile:
        if not publisher_info:
            return QoSProfile(depth=1)

        num_endpoints = len(publisher_info)
        reliability_reliable_count = 0
        durability_transient_local_count = 0

        for endpoint in publisher_info:
            profile = endpoint.qos_profile
            if profile.reliability == ReliabilityPolicy.RELIABLE:
                reliability_reliable_count += 1
            if profile.durability == DurabilityPolicy.TRANSIENT_LOCAL:
                durability_transient_local_count += 1

        request_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            liveliness=LivelinessPolicy.AUTOMATIC,
        )

        # Set reliability based on publisher offers
        if reliability_reliable_count == num_endpoints:
            request_qos.reliability = ReliabilityPolicy.RELIABLE
        else:
            if reliability_reliable_count > 0:
                logging.warning(
                    "Some, but not all, publishers are offering RELIABLE reliability. "
                    "Falling back to BEST_EFFORT as it will connect to all publishers. "
                    "Some messages from Reliable publishers could be dropped."
                )
            request_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # Set durability based on publisher offers
        if durability_transient_local_count == num_endpoints:
            request_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        else:
            if durability_transient_local_count > 0:
                logging.warning(
                    "Some, but not all, publishers are offering TRANSIENT_LOCAL durability. "
                    "Falling back to VOLATILE as it will connect to all publishers. "
                    "Previously-published latched messages will not be retrieved."
                )
            request_qos.durability = DurabilityPolicy.VOLATILE

        return request_qos

    @staticmethod
    def build_ros2_msg(
        msg_type: str | type[rclpy.node.MsgType], msg_args: Dict[str, Any]
    ) -> object:
        """Build a ROS2 message instance from string or MsgType and content dictionary."""
        if isinstance(msg_type, str):
            msg_cls = import_message_from_str(msg_type)
        else:
            msg_cls = msg_type
        msg = msg_cls()
        rosidl_runtime_py.set_message.set_message_fields(msg, msg_args)
        return msg

    @staticmethod
    def build_ros2_service_request(
        service_type: str, service_request_args: Dict[str, Any]
    ) -> Tuple[object, Type[Any]]:
        msg_cls = import_message_from_str(service_type)
        msg = msg_cls.Request()
        rosidl_runtime_py.set_message.set_message_fields(msg, service_request_args)
        return msg, msg_cls  # type: ignore

    @staticmethod
    def import_message_from_str(msg_type: str) -> Type[object]:
        msg_namespaced_type: NamespacedType = get_namespaced_type(msg_type)
        return import_message_from_namespaced_type(msg_namespaced_type)

    def get_topic_type(self, topic: str) -> str:
        names_and_types = self.node.get_topic_names_and_types(no_demangle=False)
        for name, types in names_and_types:
            if name == (topic if topic.startswith("/") else f"/{topic}"):
                if len(types) != 1:
                    raise ValueError(f"Topic {topic} has multiple types: {types}")
                return types[0]
        raise ValueError(f"Topic {topic} not found")

    @staticmethod
    def is_ros2_message(msg: Any) -> bool:
        return is_message(msg)

    @staticmethod
    def is_ros2_service(msg: Any) -> bool:
        return is_service(msg)

    @staticmethod
    def is_ros2_action(msg: Any) -> bool:
        return is_action(msg)

    @staticmethod
    def _is_nested_instance(obj: Any, nested_names: List[str]) -> bool:
        """Check if object is a nested class instance (e.g., SetBool.Request).

        Parameters
        ----------
        obj : Any
            The object to check.
        nested_names : List[str]
            List of nested class names to check for (e.g., ["Request", "Response"]).

        Returns
        -------
        bool
            True if object is a nested class instance matching one of the names.
        """
        obj_type = type(obj) if obj is not None else None
        return (
            obj_type is not None
            and hasattr(obj_type, "__qualname__")
            and any(name in obj_type.__qualname__ for name in nested_names)
            and not isinstance(obj, dict)
        )

    @staticmethod
    def extract_service_class_from_request(request: Any) -> Tuple[Type[Any], str]:
        """Extract service base class and type string from Request/Response instance.

        Uses introspection to get the parent service class from nested Request/Response
        classes. For example, from SetBool.Request() extracts SetBool class.

        Parameters
        ----------
        request : Any
            Service Request or Response instance (e.g., SetBool.Request())

        Returns
        -------
        Tuple[Type[Any], str]
            Tuple of (service_class, service_type_string)
            e.g., (SetBool, "std_srvs/srv/SetBool")

        Raises
        ------
        ValueError
            If unable to extract service class from request instance.
        """
        request_type = type(request)
        qualname = request_type.__qualname__
        module_name = request_type.__module__

        # Parse qualname to get base class name
        # ROS2 uses underscores: "SetBool_Request" -> "SetBool"
        # Also handle dot notation: "SetBool.Request" -> "SetBool"
        if "_" in qualname:
            # ROS2 format: ServiceName_Request or ServiceName_Response
            parts = qualname.split("_")
            if len(parts) >= 2 and parts[-1] in ["Request", "Response"]:
                base_class_name = "_".join(parts[:-1])
            else:
                raise ValueError(
                    f"Request/Response class {qualname} does not appear to be nested in a service class"
                )
        elif "." in qualname:
            # Dot notation: ServiceName.Request or ServiceName.Response
            base_class_name = qualname.split(".")[0]
        else:
            raise ValueError(
                f"Request/Response class {qualname} does not appear to be nested in a service class"
            )

        # Import the module and get the base class
        module = importlib.import_module(module_name)
        service_cls = getattr(module, base_class_name, None)

        if service_cls is None:
            raise ValueError(
                f"Could not find service class {base_class_name} in module {module_name}"
            )

        # Convert to service type string (e.g., "std_srvs/srv/SetBool")
        # Extract package and service name from module path
        # ROS2 service modules follow pattern: package.srv._service_name
        # e.g., "std_srvs.srv._set_bool" -> "std_srvs/srv/SetBool"
        module_parts = module_name.split(".")
        if len(module_parts) >= 2:
            package = module_parts[0]
            # Construct service type: package/srv/ServiceName
            service_type = f"{package}/srv/{base_class_name}"
        else:
            raise ValueError(
                f"Could not determine service type from module {module_name}"
            )

        return service_cls, service_type

    @staticmethod
    def extract_action_class_from_goal(goal: Any) -> Tuple[Type[Any], str]:
        """Extract action base class and type string from Goal/Result/Feedback instance.

        Uses introspection to get the parent action class from nested Goal/Result/Feedback
        classes. For example, from MoveGroup.Goal() extracts MoveGroup class.

        Parameters
        ----------
        goal : Any
            Action Goal, Result, or Feedback instance (e.g., MoveGroup.Goal())

        Returns
        -------
        Tuple[Type[Any], str]
            Tuple of (action_class, action_type_string)
            e.g., (MoveGroup, "moveit_msgs/action/MoveGroup")

        Raises
        ------
        ValueError
            If unable to extract action class from goal instance.
        """
        goal_type = type(goal)
        qualname = goal_type.__qualname__
        module_name = goal_type.__module__

        # Parse qualname to get base class name
        # ROS2 uses underscores: "NavigateToPose_Goal" -> "NavigateToPose"
        # Also handle dot notation: "MoveGroup.Goal" -> "MoveGroup"
        if "_" in qualname:
            # ROS2 format: ActionName_Goal, ActionName_Result, or ActionName_Feedback
            parts = qualname.split("_")
            if len(parts) >= 2 and parts[-1] in ["Goal", "Result", "Feedback"]:
                base_class_name = "_".join(parts[:-1])
            else:
                raise ValueError(
                    f"Goal/Result/Feedback class {qualname} does not appear to be nested in an action class"
                )
        elif "." in qualname:
            # Dot notation: ActionName.Goal, ActionName.Result, or ActionName.Feedback
            base_class_name = qualname.split(".")[0]
        else:
            raise ValueError(
                f"Goal/Result/Feedback class {qualname} does not appear to be nested in an action class"
            )

        # Import the module and get the base class
        import importlib

        module = importlib.import_module(module_name)
        action_cls = getattr(module, base_class_name, None)

        if action_cls is None:
            raise ValueError(
                f"Could not find action class {base_class_name} in module {module_name}"
            )

        # Convert to action type string (e.g., "moveit_msgs/action/MoveGroup")
        # Extract package and action name from module path
        # ROS2 action modules follow pattern: package.action._action_name
        # e.g., "moveit_msgs.action._move_group" -> "moveit_msgs/action/MoveGroup"
        module_parts = module_name.split(".")
        if len(module_parts) >= 2:
            package = module_parts[0]
            # Construct action type: package/action/ActionName
            action_type = f"{package}/action/{base_class_name}"
        else:
            raise ValueError(
                f"Could not determine action type from module {module_name}"
            )

        return action_cls, action_type

    @staticmethod
    def dict_to_message(
        msg_type: str | type[rclpy.node.MsgType], msg_dict: Dict[str, Any]
    ) -> IROS2Message:
        """Convert a dictionary to a ROS2 message class instance.

        This utility bridges LLM-oriented dict-based APIs with typed (human-friendly)
        typed message classes. LLMs can generate dicts, then convert them to
        message classes for type safety and IDE support.

        Parameters
        ----------
        msg_type : str | type[rclpy.node.MsgType]
            The ROS2 message type as string (e.g., 'geometry_msgs/msg/PoseStamped')
            or message class.
        msg_dict : Dict[str, Any]
            Dictionary containing message fields.

        Returns
        -------
        IROS2Message
            ROS2 message class instance.
        """
        return BaseROS2API.build_ros2_msg(msg_type, msg_dict)
