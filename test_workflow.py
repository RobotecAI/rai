import time

from rai.communication.ros2 import ROS2Connector, ROS2Context
from rai.communication.ros2.messages import ROS2Message
from std_srvs.srv import Trigger_Response

with ROS2Context():
    nav2_connector = ROS2Connector(node_name="Nav2Agent", namespace="rai_control")
    nav2_connector.create_service(
        service_name="navigate_to",
        on_request=lambda x, y: Trigger_Response(
            success=True, message="Navigation completed"
        ),
        service_type="std_srvs/srv/Trigger",
    )
    moveit2_connector = ROS2Connector(node_name="Moveit2Agent", namespace="rai_control")
    moveit2_connector.create_service(
        service_name="execute_trajectory",
        on_request=lambda x, y: Trigger_Response(
            success=True, message="Trajectory execution completed"
        ),
        service_type="std_srvs/srv/Trigger",
    )

    user_task_connector = ROS2Connector(
        node_name="UserTaskAgent", namespace="rai_megamind"
    )

    rai_gdino_connector = ROS2Connector(
        node_name="RAIGroundingDino", namespace="rai_perception"
    )
    rai_gdino_connector.create_service(
        service_name="grounding_dino_classify",
        on_request=lambda x, y: Trigger_Response(
            success=True, message="Detection completed"
        ),
        service_type="std_srvs/srv/Trigger",
    )
    rai_gsam_connector = ROS2Connector(
        node_name="RAIGroundedSam", namespace="rai_perception"
    )
    rai_gsam_connector.create_service(
        service_name="grounded_sam_segment",
        on_request=lambda x, y: Trigger_Response(
            success=True, message="Segmentation completed"
        ),
        service_type="std_srvs/srv/Trigger",
    )

    time.sleep(1.0)
    while True:
        user_task_connector.receive_message(source="/user_task", timeout_sec=2.0)
        time.sleep(1.0)
        user_task_connector.service_call(
            ROS2Message(payload={}),
            target="navigate_to",
            msg_type="std_srvs/srv/Trigger",
        )
        time.sleep(1.0)
        user_task_connector.service_call(
            ROS2Message(payload={}),
            target="grounding_dino_classify",
            msg_type="std_srvs/srv/Trigger",
        )
        time.sleep(1.0)
        user_task_connector.service_call(
            ROS2Message(payload={}),
            target="grounded_sam_segment",
            msg_type="std_srvs/srv/Trigger",
        )
        time.sleep(1.0)
        user_task_connector.service_call(
            ROS2Message(payload={}),
            target="execute_trajectory",
            msg_type="std_srvs/srv/Trigger",
        )
        time.sleep(1.0)
        user_task_connector.send_message(
            ROS2Message(payload={}),
            target="/user_output",
            msg_type="std_msgs/msg/String",
        )
        time.sleep(1.0)
