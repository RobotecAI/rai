import threading
import time
import rclpy
import rclpy.qos
from langchain_core.messages import HumanMessage
from rai.agents.conversational_agent import create_conversational_agent
from rai.node import RaiBaseNode
from rai.tools.ros.manipulation import GetObjectPositionsTool, MoveToPointTool
from rai.tools.ros.native import GetCameraImage, Ros2GetTopicsNamesAndTypesTool
from rai.utils.model_initialization import get_llm_model
from rai.communication.ros2.connectors import ROS2ARIConnector
from rai.tools.ros2.topics import GetROS2TopicsNamesAndTypesTool, GetROS2ImageTool
from rai_open_set_vision.tools import GetGrabbingPointTool

rclpy.init()
connector = ROS2ARIConnector(node_name="cosik")
node = connector.node
node.declare_parameter("conversion_ratio", 1.0)

tool = GetObjectPositionsTool(
    connector=connector,
    target_frame="panda_link0",
    source_frame="RGBDCamera5",
    camera_topic="/color_image5",
    depth_topic="/depth_image5",
    camera_info_topic="/color_camera_info5",
    get_grabbing_point_tool=GetGrabbingPointTool(connector=connector),
)

response = tool._run(object_name="carrot")
time.sleep(10)
print(response)
