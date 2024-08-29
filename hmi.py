import threading
import time
from queue import Queue

import rclpy
from pydantic import BaseModel
from rclpy.action.client import ActionClient
from rclpy.node import Node

from rai_interfaces.action import Task as TaskAction


class Task(BaseModel):
    name: str
    description: str


class MyNode(Node):
    def __init__(self, queue: Queue) -> None:
        super().__init__("my_node")
        self.task_action_client = ActionClient(self, TaskAction, "perform_task")
        self.queue = queue
        self.task_running = False

    def task_loop(self):
        self.get_logger().info("Task loop started")
        while True:
            self.get_logger().info(f"Checkfng tasks: {self.queue.qsize()}")
            if self.task_running or self.queue.empty():
                time.sleep(1)
                continue
            task = self.queue.get()
            self.task_running = True
            self.get_logger().info(f"Received task: {task}")
            self.add_task_to_queue(task)

    def add_task_to_queue(self, task: Task):
        """Sends a task to the action server to be handled by the rai node."""

        if not self.task_action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Task action server not available!")
            raise Exception("Task action server not available!")

        goal_msg = TaskAction.Goal()
        goal_msg.task = task.name
        goal_msg.description = task.description

        self.get_logger().info(f"Sending task to action server: {goal_msg.task}")
        self._send_goal_future = self.task_action_client.send_goal_async(
            goal_msg, feedback_callback=self.task_feedback_callback
        )
        self._send_goal_future.add_done_callback(self.task_goal_response_callback)

    def task_goal_response_callback(self, future):
        """Callback for handling the response from the action server when the goal is sent."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Task goal rejected by action server.")
            self.task_running = False
            return

        self.get_logger().info("Task goal accepted by action server.")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.task_result_callback)

    def task_result_callback(self, future):
        """Callback for handling the result from the action server."""
        result = future.result().result
        self.task_running = False
        if result.success:
            self.get_logger().info(f"Task completed successfully: {result.report}")
        else:
            self.get_logger().error(f"Task failed: {result}")

    def task_feedback_callback(self, feedback_msg):
        """Callback for receiving feedback from the action server."""
        self.get_logger().info(f"Task feedback received: {feedback_msg.feedback}")


if __name__ == "__main__":
    rclpy.init()

    q = Queue()
    node = MyNode(q)

    threading.Thread(target=node.task_loop).start()
    print("threads started")
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print("threads started")
    time.sleep(2)

    print("Added task to queue")
    q.put(Task(name="Task 1", description="Task 1 description"))

    q.put(Task(name="Task 2", description="Task 2 description"))
    q.put(Task(name="Task 3", description="Task 3 description"))
    while True:
        pass

    rclpy.shutdown()
