# File specifically for ROS actions
import logging
import os
import subprocess
from typing import TYPE_CHECKING

import rclpy
from std_srvs.srv import SetBool

from .actions import Action

if TYPE_CHECKING:
    from rai.scenario_engine.scenario_engine import ScenarioRunner


class StopRobotAction(Action):
    def run(self, runner: "ScenarioRunner") -> None:

        rclpy.init()
        node = rclpy.create_node("stop_robot_node")  # type: ignore
        client = node.create_client(SetBool, "/safety_stop")  # type: ignore

        request = SetBool.Request()
        request.data = False

        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)

        if future.result() is not None:
            if future.result().success:  # type: ignore
                logging.critical("Robot stopped successfully")
            else:
                logging.error("Failed to stop the robot")
        else:
            logging.error("Service call failed")

        node.destroy_node()
        rclpy.shutdown()


class RosAPICallAction(Action):
    def run(self, runner: "ScenarioRunner") -> None:
        def wrap_for_ros_api(action: str):
            source_command = "source /opt/ros/humble/setup.bash"
            return f"{source_command} && ROS_DOMAIN_ID={os.getenv('ROS_DOMAIN_ID')} {action}"

        action = (
            runner.history[-1]
            .content.replace("```", "")
            .replace("bash", "")
            .replace("sh", "")
            .replace("\n", "")
        )
        action = wrap_for_ros_api(action)
        try:
            self.logger.info(f"Running action: {action}")
            output = subprocess.run(
                ["/bin/bash", "-c", action],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout.decode("utf-8")

            self.logger.info(f"Action output: \n{output}\n")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Action failed: {e}")
