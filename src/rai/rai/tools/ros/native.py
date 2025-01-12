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


import importlib
import json
import time
from typing import Annotated, Any, Dict, List, OrderedDict, Tuple, Type

import rclpy
import rclpy.callback_groups
import rclpy.executors
import rclpy.node
import rclpy.qos
import rclpy.subscription
import rclpy.task
import rosidl_runtime_py.set_message
import rosidl_runtime_py.utilities
import sensor_msgs.msg
from langchain.tools import tool
from langchain_core.tools import InjectedToolArg
from rclpy.client import Client
from rosidl_runtime_py import message_to_ordereddict
from rosidl_runtime_py.utilities import get_namespaced_type

from rai.node import RaiBaseNode
from rai.utils.ros_async import get_future_result

from .utils import convert_ros_img_to_base64, get_transform, import_message_from_str


@tool
def ros2_get_topics_names_and_types(
    node: Annotated[rclpy.node.Node, InjectedToolArg]
) -> List[Tuple[str, List[str]]]:
    """A tool for getting all ros2 topics names and types"""
    return node.get_topic_names_and_types()


@tool
def ros2_get_robot_interfaces(
    node: Annotated[RaiBaseNode, InjectedToolArg]
) -> Dict[str, Any]:
    """A tool for getting all ros2 robot interfaces: topics, services and actions"""
    return node.ros_discovery_info.dict()


@tool
def ros2_show_msg_interface(msg_name: str) -> str:
    """
    Show ros2 message interface in json format as string.

    Parameters
    ----------
    msg_name : str
        The name of the message. For example "geometry_msgs/msg/PoseStamped"
    """
    msg_cls: Type = rosidl_runtime_py.utilities.get_interface(msg_name)
    try:
        msg_dict: OrderedDict = rosidl_runtime_py.convert.message_to_ordereddict(
            msg_cls()
        )
        return json.dumps(msg_dict)
    except NotImplementedError:
        # For action classes that can't be instantiated
        goal_dict: OrderedDict = rosidl_runtime_py.convert.message_to_ordereddict(
            msg_cls.Goal()
        )

        result_dict: OrderedDict = rosidl_runtime_py.convert.message_to_ordereddict(
            msg_cls.Result()
        )

        feedback_dict: OrderedDict = rosidl_runtime_py.convert.message_to_ordereddict(
            msg_cls.Feedback()
        )
        return json.dumps(
            {"goal": goal_dict, "result": result_dict, "feedback": feedback_dict}
        )


# TODO(boczekbartek): use few shot from langchain docs: https://python.langchain.com/docs/how_to/tools_few_shot/
@tool
def ros2_pub_msg(
    node: Annotated[RaiBaseNode, InjectedToolArg],
    topic_name: str,
    msg_type: str,
    msg_args: Dict[str, Any],
    rate: int = 10,
    timeout_sec: int = 1,
) -> None:
    """A tool for publishing a message to a ros2 topic

    By default 10 messages are published for 1 second. If you want to publish multiple messages, you can specify 'rate' and 'timeout_sec'.

    Parameters
    ----------
    topic_name : str
        The name of the topic to publish the message
    msg_type : str
        The type of the message
    msg_args : Dict[str, Any]
        The arguments of the message
    rate : int, default=10
        The rate at which to publish the message
    timeout_sec : int, default=1
        The timeout in seconds

    Example usage:

    ```python
    tool = Ros2PubMessageTool()
    tool.run(
        {
            "topic_name": "/some_topic",
            "msg_type": "geometry_msgs/Point",
            "msg_args": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rate" : 10,
            "timeout_sec" : 1
        }
    )
    ```
    """

    if "/msg/" not in msg_type:
        raise ValueError("msg_name must contain 'msg' in the name.")

    msg_cls: Type = import_message_from_str(msg_type)
    msg = msg_cls()
    rosidl_runtime_py.set_message.set_message_fields(msg, msg_args)

    publisher = node.create_publisher(
        msg_cls, topic_name, 10, callback_group=node.callback_group
    )

    def callback():
        publisher.publish(msg)
        node.get_logger().info(f"Published message '{msg}' to topic '{topic_name}'")

    ts = time.perf_counter()
    timer = node.create_timer(1.0 / rate, callback, callback_group=node.callback_group)

    while time.perf_counter() - ts < timeout_sec:
        time.sleep(0.1)

    timer.cancel()
    timer.destroy()

    node.get_logger().info(
        f"Published messages for {timeout_sec}s to topic '{topic_name}' with rate {rate}"
    )


@tool(response_format="content_and_artifact")
def ros2_get_msg_from_topic(
    node: Annotated[RaiBaseNode, InjectedToolArg], topic_name: str
) -> Tuple[str, Dict[str, Any]]:
    """A tool for getting a message from a ros2 topic

    Parameters
    ----------
    topic_name : str
        The name of the topic to get the message from
    """
    msg = node.get_raw_message_from_topic(topic_name)

    if type(msg) is sensor_msgs.msg.Image:
        img = convert_ros_img_to_base64(msg)
        return "Got image", {"images": [img]}
    else:
        return str(msg), {}


@tool
def ros2_service_caller(
    node: Annotated[rclpy.node.Node, InjectedToolArg],
    service_name: str,
    service_type: str,
    request_args: Dict[str, Any],
    timeout_sec: int = 1,
) -> str:
    """A tool for calling any ROS2 service dynamically

    Parameters
    ----------
    service_name : str
        The name of the service to call
    service_type : str
        The type of the service in typical ros2 format
    request_args : Dict[str, Any]
        The arguments for the service request
    timeout_sec : int, default=1
        Request timeout.
    """
    if not service_name.startswith("/"):
        service_name = f"/{service_name}"

    try:
        srv_module, _, srv_name = service_type.split("/")
        srv_class = getattr(importlib.import_module(f"{srv_module}.srv"), srv_name)
        request = srv_class.Request()
        rosidl_runtime_py.set_message.set_message_fields(request, request_args)
    except Exception as e:
        return f"Failed to build service request: {e}"

    namespaced_type = get_namespaced_type(service_type)
    client: Client = node.create_client(
        rosidl_runtime_py.import_message.import_message_from_namespaced_type(
            namespaced_type
        ),
        service_name,
    )

    if not client.wait_for_service(timeout_sec=1.0):
        return f"Service '{service_name}' is not available"

    future = client.call_async(request)
    result = get_future_result(future, timeout_sec=timeout_sec)

    client.destroy()

    if result is not None:
        return str(result)
    else:
        return f"Service call to '{service_name}' failed"


@tool(response_format="content_and_artifact")
def ros2_get_camera_image(
    node: Annotated[RaiBaseNode, InjectedToolArg], topic_name: str
) -> Tuple[str, Dict[str, Any]]:
    """A tool for getting an image from a ros2 camera

    Parameters
    ----------
    topic_name : str
        The name of the topic to get the image from
    """
    msg = node.get_raw_message_from_topic(topic_name)
    img = convert_ros_img_to_base64(msg)
    return "Got image", {"images": [img]}


@tool
def ros2_get_transform(
    node: Annotated[rclpy.node.Node, InjectedToolArg],
    target_frame: str = "map",
    source_frame: str = "body_link",
) -> dict:
    """Get tf from source_frame to target_frame"""
    return message_to_ordereddict(get_transform(node, target_frame, source_frame))
