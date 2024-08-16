from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String

from rai_hmi.base import BaseHMINode


class VoiceHMINode(BaseHMINode):
    def __init__(self, node_name: str, robot_description_package: str):
        super().__init__(node_name, robot_description_package)

        self.callback_group = ReentrantCallbackGroup()
        self.hmi_subscription = self.create_subscription(
            String,
            "from_human",
            self.handle_human_message,
            10,
            callback_group=self.callback_group,
        )

        self.hmi_publisher = self.create_publisher(
            String, "to_human", 10, callback_group=self.callback_group
        )

    def handle_human_message(self, msg: String):
        self.processing = True

        # handle human message
        output = ""  # self.agent(msg.data, config=config)

        self.processing = False
        self.hmi_publisher.publish(String(data=output))

    def handle_feedback_request(self, feedback_query: str) -> str:
        self.processing = True

        # handle feedback request
        output = ""  # self.agent(feedback_query, config=config)

        self.processing = False
        return output
