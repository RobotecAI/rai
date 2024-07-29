import logging
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import rclpy
import rclpy.callback_groups
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
    def get_uids(self) -> List[str]:
        pass

    @abstractmethod
    def get_results(self, uid: Optional[str] = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def cancel_action(self, uid: str) -> bool:
        pass


@dataclass
class RaiActionStoreRecord:
    uid: str
    action_name: str
    action_type: str
    action_goal_args: Dict[str, Any]
    result_future: rclpy.Future


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
        result_future: rclpy.Future,
    ):
        self._actions[uid] = RaiActionStoreRecord(
            uid=uid,
            action_name=action_name,
            action_type=action_type,
            action_goal_args=action_goal_args,
            result_future=result_future,
        )

    def get_uids(self) -> List[str]:
        # TODO(boczekbartek): optimize it
        return list(self._actions.keys())

    def cancel_action(self, uid: str) -> bool:
        if uid not in self._actions:
            raise KeyError(f"Unknown action: {uid=}")
        self._actions[uid].result_future.cancel()
        self._actions.pop(uid, None)
        logging.getLogger().info(f"Action(uid={uid}) canceled")
        return True

    def get_results(self, uid: Optional[str] = None) -> Dict[str, Any]:
        results = dict()
        done_actions = list()
        if uid is not None:
            uids = [uid]
        else:
            uids = self.get_uids()
        for uid in uids:
            if uid not in self._actions:
                raise KeyError(f"Unknown action: {uid=}")
            action = self._actions[uid]
            done = action.result_future.done()
            logging.getLogger().debug(f"Action(uid={uid}) done: {done}")
            if done:
                results[uid] = action.result_future.result()
                done_actions.append(uid)
            else:
                results[uid] = "Not done yet"

        # Remove done actions
        logging.getLogger().info(f"Removing done actions: {results.keys()=}")
        for uid in done_actions:
            self._actions.pop(uid, None)

        return results


class RaiNode(Node):
    def __init__(self):
        super().__init__("rai_node")

        self.callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        self._actions_cache = RaiActionStore()

    def get_actions_cache(self) -> RaiActionStoreInterface:
        return self._actions_cache

    def get_results(self, uid: Optional[str]):
        self.get_logger().info("Getting results")
        return self._actions_cache.get_results(uid)

    def cancel_action(self, uid: str):
        self.get_logger().info(f"Canceling action: {uid=}")
        return self._actions_cache.cancel_action(uid)

    def get_running_actions(self, uid: str):
        self.get_logger().info(f"Getting running actions: {uid=}")
        return self._actions_cache.get_uids()

    def feedback_callback(self, uid: str, feedback_msg: Any):
        feedback = feedback_msg.feedback
        self.get_logger().info(f"Received feedback: {feedback}")
