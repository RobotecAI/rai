import functools
from abc import abstractmethod

from pydantic import UUID4
from rclpy.action import ActionClient, ActionServer
from rclpy.node import Node

from rai_hmi.task import Task
from rai_interfaces.action import Task as TaskAction
from rai_interfaces.action import TaskFeedback


class TaskActionMixin(Node):
    """
    Mixin class to handle Task action client and TaskFeedback action server.

    Provides methods to:
    - Send a task to the action server.
    - Handle feedback from the action server.
    - Handle task result responses.
    - Implement an action server for providing task feedback.

    Abstract Methods:
        handle_task_feedback_request: Must be implemented by subclasses to process task feedback requests.
    """

    def initialize_task_action_client_and_server(self):
        """Initialize the action client and server for task handling."""
        self.task_action_client = ActionClient(self, TaskAction, "perform_task")
        self.task_feedback_action_server = ActionServer(
            self, TaskFeedback, "provide_task_feedback", self.handle_task_feedback
        )

    def add_task_to_queue(self, task: Task):
        """Sends a task to the action server to be handled by the rai node."""

        if not self.task_action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Task action server not available!")
            raise Exception("Task action server not available!")

        # Register task
        self.task_results[task.uid] = None
        self.task_running[task.uid] = False

        # Create ros2 action goal
        goal_msg = TaskAction.Goal()
        goal_msg.task = task.name
        goal_msg.description = task.description
        goal_msg.priority = task.priority

        self.get_logger().info(f"Sending task to action server: {goal_msg.task}")

        feedback_callback = functools.partial(self.task_feedback_callback, uid=task.uid)
        self._send_goal_future = self.task_action_client.send_goal_async(
            goal_msg, feedback_callback=feedback_callback
        )

        goal_response_callback = functools.partial(
            self.task_goal_response_callback, uid=task.uid
        )
        self._send_goal_future.add_done_callback(goal_response_callback)

    def task_goal_response_callback(self, future, uid: UUID4):
        """Callback for handling the response from the action server when the goal is sent."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Task goal rejected by action server.")
            self.task_running[uid] = False
            return

        self.task_running[uid] = True
        self.get_logger().info("Task goal accepted by action server.")
        self._get_result_future = goal_handle.get_result_async()

        done_callback = functools.partial(self.task_result_callback, uid=uid)
        self._get_result_future.add_done_callback(done_callback)

    def task_feedback_callback(self, feedback_msg, uid: UUID4):
        """Callback for receiving feedback from the action server."""
        self.get_logger().info(f"Task feedback received: {feedback_msg.feedback}")
        self.task_feedbacks.put((uid, feedback_msg.feedback))

    def task_result_callback(self, future, uid: UUID4):
        """Callback for handling the result from the action server."""
        result = future.result().result
        self.task_running[uid] = False
        if result.success:
            self.get_logger().info(f"Task completed successfully: {result.report}")
            self.task_results[uid] = result
        else:
            self.get_logger().error(f"Task failed: {result.result_message}")
            self.task_results[uid] = "ERROR"

    @abstractmethod
    def handle_task_feedback_request(self, goal_handle):
        """Abstract method to handle TaskFeedback action request."""

    def handle_task_feedback(self, goal_handle):
        """Handles the TaskFeedback action request."""
        return self.handle_task_feedback_request(goal_handle)
