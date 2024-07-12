
from langchain_aws import ChatBedrock
from typing import List

from .task import Task

from langchain_core.messages import (
  BaseMessage,
  SystemMessage,
  HumanMessage,
  AIMessage,
  ToolMessage
)

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


SYSTEM_PROMPT = """
You are a helpful office assistant robot. You driver around using a Husarion ROSBot XL platform, equipped with
1) a front-facing camera,
2) a single-ray, 360° horizontal FOV LiDAR,
3) an audio interface, consisting of a microphone and a speaker,
4) a container box where small items can be placed.

You drive around the office performing various tasks, and people might talk to you through an audio interface.
If a user requests a task from you, continue the conversation until you are sure you have all information
required to perform the task.

Assume the conversation is carried over a voice interface, so try not to overwhelm the user.
If you have multiple questions, please ask them one by one allowing user to respond before
moving forward to the next question.

Once you have gathered all information, add this task to queue, and acknoledge it to the user.
"""


class HMINode(Node):
    def __init__(self):
        super().__init__("rai_hmi_node")

        # TODO: add parameter to choose model

        self.history: List[BaseMessage] = [SystemMessage(SYSTEM_PROMPT)]

        llm = ChatBedrock(
            model_id='anthropic.claude-3-5-sonnet-20240620-v1:0',
            region_name='us-east-1'
        )

        tools = [self.add_task_to_queue]
        self.llm_with_tools = llm.bind_tools(tools)

        self.hmi_subscription = self.create_subscription(
            String, "from_human", self.handle_human_message, 10
        )

        self.hmi_publisher = self.create_publisher(
            String, "to_human", 10
        )

        self.task_addition_request_publisher = self.create_publisher(
            String, "tasks_addition_requests", 10
        )

        self.get_logger().info("HMI Node has been started")

    def handle_human_message(self, human_ros_msg: String):
        self.history.append(HumanMessage(human_ros_msg.data))

        ai_msg: AIMessage = self.llm_with_tools.invoke(self.history)
        self.history.append(ai_msg)

        # TODO: Make this code safe (handle exceptions when parsing)
        for tool_call in ai_msg.tool_calls:
            task_json = tool_call["args"]["task"]
            task = Task.model_validate(task_json)
            self.add_task_to_queue(task)

            self.history.append(
                ToolMessage("Task added!", tool_call_id=tool_call["id"])
            )

        if ai_msg.content:
            self.send_message_to_human(ai_msg.content)
        elif ai_msg.tool_calls:
            # TODO: This is ugly code repetition and should be refactored
            ai_msg: AIMessage = self.llm_with_tools.invoke(self.history)
            self.history.append(ai_msg)

            if ai_msg.content:
              self.send_message_to_human(ai_msg.content)

    def send_message_to_human(self, content: str):
        msg = String()
        msg.data = content
        self.hmi_publisher.publish(msg)

    def add_task_to_queue(self, task: Task):
        msg = String()
        msg.data = task.model_dump_json()
        self.task_addition_request_publisher.publish(msg)

    # TODO: This method is currently not called anywhere
    def clear_history(self):
        # Remove everything except system prompt
        self.history = self.history[:1]


def main():
    rclpy.init()
    hmi_node = HMINode()
    rclpy.spin(hmi_node)
    hmi_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
