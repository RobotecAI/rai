import logging
import os
import time

from rai.agents.base import BaseAgent
from rai.communication.ros2 import ROS2Connector, ROS2Context
from rai.communication.ros2.messages import ROS2Message
from std_srvs.srv import Trigger_Response

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


endpoint = os.getenv("RAI_OBS_ENDPOINT")


class DemoAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="DemoAgent", observability_endpoint=endpoint)
        self.nav2_connector = ROS2Connector(node_name="Nav2Agent")
        self.moveit2_connector = ROS2Connector(node_name="Moveit2Agent")
        self.user_task_connector = ROS2Connector(node_name="UserTaskAgent")
        self.rai_gdino_connector = ROS2Connector(node_name="RAIGroundingDino")
        self.rai_gsam_connector = ROS2Connector(node_name="RAIGroundedSam")

    def setup(self):
        _ = self.nav2_connector.create_service(
            service_name="navigate_to",
            on_request=lambda x, y: Trigger_Response(
                success=True, message="Navigation completed"
            ),
            service_type="std_srvs/srv/Trigger",
        )
        _ = self.moveit2_connector.create_service(
            service_name="execute_trajectory",
            on_request=lambda x, y: Trigger_Response(
                success=True, message="Trajectory execution completed"
            ),
            service_type="std_srvs/srv/Trigger",
        )
        _ = self.rai_gdino_connector.create_service(
            service_name="grounding_dino_classify",
            on_request=lambda x, y: Trigger_Response(
                success=True, message="Detection completed"
            ),
            service_type="std_srvs/srv/Trigger",
        )
        _ = self.rai_gsam_connector.create_service(
            service_name="grounded_sam_segment",
            on_request=lambda x, y: Trigger_Response(
                success=True, message="Segmentation completed"
            ),
            service_type="std_srvs/srv/Trigger",
        )

    def run(self):
        pass

    def stop(self):
        pass


with ROS2Context():
    agent = DemoAgent()
    agent.setup()

    time.sleep(1.0)
    for i in range(10):
        _ = agent.user_task_connector.receive_message(
            source="/user_task", timeout_sec=2.0
        )
        _ = agent.user_task_connector.service_call(
            ROS2Message(payload={}),
            target="navigate_to",
            msg_type="std_srvs/srv/Trigger",
        )
        _ = agent.user_task_connector.service_call(
            ROS2Message(payload={}),
            target="grounding_dino_classify",
            msg_type="std_srvs/srv/Trigger",
        )
        _ = agent.user_task_connector.service_call(
            ROS2Message(payload={}),
            target="grounded_sam_segment",
            msg_type="std_srvs/srv/Trigger",
        )
        _ = agent.user_task_connector.service_call(
            ROS2Message(payload={}),
            target="execute_trajectory",
            msg_type="std_srvs/srv/Trigger",
        )
        _ = agent.user_task_connector.send_message(
            ROS2Message(payload={}),
            target="/user_output",
            msg_type="std_msgs/msg/String",
        )
