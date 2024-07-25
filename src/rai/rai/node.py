import functools
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List

import rclpy
from langchain_core.pydantic_v1 import BaseModel, Field
from rclpy.action.client import ClientGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node


class RaiActionStoreInterface(metaclass=ABCMeta):
    @abstractmethod
    def register_action(
        self,
        uid: str,
        action_name: str,
        action_type: str,
        action_goal_args: Dict[str, Any],
    ):
        pass

    @abstractmethod
    def add_result(self, uid: str, result: Any):
        pass

    @abstractmethod
    def add_feedback(self, uid: str, feedback: Any):
        pass


class RaiActionStoreRecord(BaseModel):
    # TODO(boczekbartek): temporary, because of cicular dependency - needs to be refactored
    uid: str = Field(..., description="Unique id")
    action_name: str = Field(..., description="Name of the action")
    action_type: str = Field(..., description="Type of the action")
    action_goal_args: Dict[str, Any] = Field(
        ..., description="Arguments for the action goal"
    )


class RaiActionStore(RaiActionStoreInterface):
    def __init__(self) -> None:
        self._actions: Dict[str, RaiActionStoreRecord] = dict()
        self._results: Dict[str, Any] = dict()
        self._feedbacks: Dict[str, List[Any]] = defaultdict(list)

    def register_action(
        self,
        uid: str,
        action_name: str,
        action_type: str,
        action_goal_args: Dict[str, Any],
    ):
        self._actions[uid] = RaiActionStoreRecord(
            uid=uid,
            action_name=action_name,
            action_type=action_type,
            action_goal_args=action_goal_args,
        )

    def add_action(self, uid: str):
        self._actions[uid]

    def add_result(self, uid: str, result: Any):
        self._results[uid] = result

    def add_feedback(self, uid: str, feedback: Any):
        self._feedbacks[uid].append(feedback)

    def get_results(self, drop: bool = True) -> Dict[str, Any]:
        results = self._results.copy()
        if drop:
            for uid in results.keys():
                self._feedbacks.pop(uid, None)
                self._actions.pop(uid, None)
            self._results.clear()

        return results


class RaiNode(Node):
    def __init__(self):
        super().__init__("rai_node")

        self.callback_group = ReentrantCallbackGroup()
        self._actions_cache = RaiActionStore()

    def get_actions_cache(self) -> RaiActionStoreInterface:
        return self._actions_cache

    def get_results(self) -> Dict[str, Any]:
        self.get_logger().info("Getting results")
        return self._actions_cache.get_results(drop=False)
        # return self._actions_cache._feedbacks

    # Calllback names follow official ros2 actions tutorial
    def goal_response_callback(self, uid: str, future: rclpy.Future):
        goal_handle: ClientGoalHandle = future.result()  # type: ignore
        if not goal_handle.accepted:
            self.get_actions_cache().add_result(uid, "Action rejected")
            return
        self.get_logger().info("Goal accepted")

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(
            functools.partial(self.get_result_callback, uid)
        )

    def get_result_callback(self, uid: str, future: rclpy.Future):
        result = future.result()
        self.get_logger().info(f"Received result: {result}")
        self.get_actions_cache().add_result(uid, result)

    def feedback_callback(self, uid: str, feedback_msg: Any):
        feedback = feedback_msg.feedback
        self.get_logger().info(f"Received feedback: {feedback}")
        self.get_actions_cache().add_feedback(uid, feedback)
