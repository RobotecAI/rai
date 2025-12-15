# Copyright (C) 2025 Julia Jia
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

import logging

from rclpy.node import Node


class ROS2LogHandler(logging.Handler):
    """Log handler that forwards Python logging to ROS2 logger."""

    def __init__(self, ros2_node: Node):
        super().__init__()
        self.ros2_node = ros2_node

    def emit(self, record):
        log_msg = self.format(record)
        if record.levelno >= logging.ERROR:
            self.ros2_node.get_logger().error(log_msg)
        elif record.levelno >= logging.WARNING:
            self.ros2_node.get_logger().warning(log_msg)
        elif record.levelno >= logging.INFO:
            self.ros2_node.get_logger().info(log_msg)
        else:
            self.ros2_node.get_logger().debug(log_msg)
