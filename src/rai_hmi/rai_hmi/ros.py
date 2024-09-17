import threading
from queue import Queue
from typing import Optional, Tuple

import rclpy
from rclpy.executors import MultiThreadedExecutor

from rai.node import RaiBaseNode
from rai_hmi.base import BaseHMINode


def initialize_ros_nodes(
    _feedbacks_queue: Queue, robot_description_package: Optional[str]
) -> Tuple[BaseHMINode, RaiBaseNode]:
    rclpy.init()

    hmi_node = BaseHMINode(
        f"{robot_description_package}_hmi_node",
        queue=_feedbacks_queue,
        robot_description_package=robot_description_package,
    )

    # TODO(boczekbartek): this node shouldn't be required to initialize simple ros2 tools
    rai_node = RaiBaseNode(node_name="__rai_node__")

    executor = MultiThreadedExecutor()
    executor.add_node(hmi_node)
    executor.add_node(rai_node)

    threading.Thread(target=executor.spin, daemon=True).start()

    return hmi_node, rai_node
