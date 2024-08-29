import time

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from rai_interfaces.action import Task


class MyNode(Node):
    def __init__(self) -> None:
        super().__init__("my_node_server")
        self.task_action_server = ActionServer(
            self, Task, "perform_task", self.execute_callback
        )

    def execute_callback(self, goal_handle):
        self.get_logger().info(f"Executing goal: {goal_handle.request.description} ")

        feedback_msg = Task.Feedback()
        feedback_msg.current_status = "Not started"

        for i in range(5):
            feedback_msg.current_status = "In progress"
            self.get_logger().info("Feedback: {0}".format(feedback_msg.current_status))
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()

        result = Task.Result()
        result.success = True
        result.report = "Task completed successfully"
        self.get_logger().info("Result: {0}".format(result.success))
        self.get_logger().info("Report: {0}".format(result.report))

        return result


if __name__ == "__main__":
    rclpy.init()
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()
