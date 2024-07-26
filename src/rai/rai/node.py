import logging
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import rclpy
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
        result_future: rclpy.Future,
    ):
        pass

    @abstractmethod
    def add_result(self, uid: str, result: Any):
        pass

    @abstractmethod
    def add_feedback(self, uid: str, feedback: Any):
        pass


@dataclass
class RaiActionStoreRecord:
    # TODO(boczekbartek): temporary, because of cicular dependency - needs to be refactored
    uid: str
    action_name: str
    action_type: str
    action_goal_args: Dict[str, Any]
    result_future: rclpy.Future


class RaiActionStore(RaiActionStoreInterface):
    def __init__(self) -> None:
        self._actions: List[RaiActionStoreRecord] = list()
        self._results: Dict[str, Any] = dict()
        self._feedbacks: Dict[str, List[Any]] = defaultdict(list)

    def register_action(
        self,
        uid: str,
        action_name: str,
        action_type: str,
        action_goal_args: Dict[str, Any],
        result_future: rclpy.Future,
    ):
        self._actions.append(
            RaiActionStoreRecord(
                uid=uid,
                action_name=action_name,
                action_type=action_type,
                action_goal_args=action_goal_args,
                result_future=result_future,
            )
        )

    def add_result(self, uid: str, result: Any):
        self._results[uid] = result

    def add_feedback(self, uid: str, feedback: Any):
        self._feedbacks[uid].append(feedback)

    def get_results(self) -> Dict[str, Any]:
        results = dict()
        to_drop = list()

        # Get results for done actions
        for i, a in enumerate(self._actions):
            done = a.result_future.done()
            logging.getLogger().debug(f"Action(uid={a.uid}) done: {done}")
            if done:
                results[a.uid] = a.result_future.result()
                to_drop.append(i)

        # Remove done actions
        self.actions = [a for i, a in enumerate(self._actions) if i not in to_drop]

        return results


class RaiNode(Node):
    def __init__(self):
        super().__init__("rai_node")

        self.callback_group = ReentrantCallbackGroup()
        self._actions_cache = RaiActionStore()

    def get_actions_cache(self) -> RaiActionStoreInterface:
        return self._actions_cache

    def get_results(self):
        self.get_logger().info("Getting results")
        return self._actions_cache.get_results()

    def get_result_callback(self, uid: str, future: rclpy.Future):
        result = future.result()
        self.get_logger().info(f"Received result: {result}")
        self.get_actions_cache().add_result(uid, result)

    def feedback_callback(self, uid: str, feedback_msg: Any):
        feedback = feedback_msg.feedback
        self.get_logger().info(f"Received feedback: {feedback}")
        self.get_actions_cache().add_feedback(uid, feedback)
