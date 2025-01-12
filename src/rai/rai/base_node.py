from typing import Annotated, Any, Dict, List, Optional, Union

import rclpy.callback_groups
from langchain.tools import tool
from langchain_core.tools import InjectedToolArg
from rclpy.node import Node

from rai.ros2_apis import Ros2ActionsAPI, Ros2TopicsAPI
from rai.utils.ros import NodeDiscovery
from rai.utils.ros_executors import MultiThreadedExecutorFixed


class RaiBaseNode(Node):
    def __init__(
        self,
        allowlist: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.callback_group = rclpy.callback_groups.ReentrantCallbackGroup()

        self.ros_discovery_info = NodeDiscovery(self, allowlist=allowlist)
        self.async_action_client = Ros2ActionsAPI(self)
        self.topics_handler = Ros2TopicsAPI(
            self, self.callback_group, self.ros_discovery_info
        )
        self.ros_discovery_info.add_setter(self.topics_handler.set_ros_discovery_info)

    # ------------- ros2 topics interface -------------
    def get_raw_message_from_topic(self, topic: str, timeout_sec: int = 5) -> Any:
        return self.topics_handler.get_raw_message_from_topic(topic, timeout_sec)

    # ------------- ros2 actions interface -------------
    def run_action(
        self, action_name: str, action_type: str, action_goal_args: Dict[str, Any]
    ):
        return self.async_action_client.run_action(
            action_name, action_type, action_goal_args
        )

    def get_task_result(self) -> str:
        return self.async_action_client.get_task_result()

    def is_task_complete(self) -> bool:
        return self.async_action_client.is_task_complete()

    @property
    def action_feedback(self) -> Any:
        return self.async_action_client.action_feedback

    def cancel_task(self) -> Union[str, bool]:
        return self.async_action_client.cancel_task()

    # ------------- other methods -------------
    def spin(self):
        executor = MultiThreadedExecutorFixed()
        executor.add_node(self)
        executor.spin()


@tool
def ros2_run_action_async(
    node: Annotated[RaiBaseNode, InjectedToolArg],
    action_name: str,
    action_type: str,
    action_goal_args: Dict[str, Any],
) -> str:
    """A generic tool for sending goal of ros2 action"""
    return node.run_action(action_name, action_type, action_goal_args)


@tool
def ros2_is_action_complete(node: Annotated[RaiBaseNode, InjectedToolArg]) -> bool:
    """A tool for checking the result of submitted ros2 action"""
    return node.is_task_complete()


@tool
def ros2_get_action_result(node: Annotated[RaiBaseNode, InjectedToolArg]) -> str:
    """A tool for checking the result of submitted ros2 action"""
    return node.get_task_result()


@tool
def ros2_cancel_action(
    node: Annotated[RaiBaseNode, InjectedToolArg]
) -> Union[str, bool]:
    """Cancel submitted ros2 action"""
    return node.cancel_task()


@tool
def ros2_get_action_feedback(node: Annotated[RaiBaseNode, InjectedToolArg]) -> str:
    """A tool for checking the feedback of submitted ros2 action"""
    return str(node.action_feedback)
