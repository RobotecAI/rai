from typing import List

import rclpy
from langchain_aws import ChatBedrock
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .task import Task


class HMINode(Node):
    def __init__(self):
        super().__init__("rai_hmi_node")

        # TODO: add parameter to choose model

        self.history: List[BaseMessage] = []

        llm = ChatBedrock(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            region_name="us-east-1",
        )

        tools = [self.add_task_to_queue]
        self.llm_with_tools = llm.bind_tools(tools)

        self.hmi_subscription = self.create_subscription(
            String, "from_human", self.handle_human_message, 10
        )

        self.hmi_publisher = self.create_publisher(String, "to_human", 10)

        self.task_addition_request_publisher = self.create_publisher(
            String, "task_addition_requests", 10
        )

        # TODO
        # self.documentation_service = self.create_client(
        #     VectorStoreRetrieval,
        #     "rai_whoami_documentation_service",
        #     self.documentation_service_callback,
        # )
        self.constitution_service = self.create_client(
            Trigger,
            "rai_whoami_constitution_service",
        )
        self.identity_service = self.create_client(
            Trigger, "rai_whoami_identity_service"
        )

        self.get_logger().info("HMI Node has been started")
        self.initialize_system_prompt()

    def initialize_system_prompt(self):
        while not self.constitution_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Constitution service not available, waiting again..."
            )

        while not self.identity_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Identity service not available, waiting again...")

        constitution_request = Trigger.Request()

        constitution_future = self.constitution_service.call_async(constitution_request)
        rclpy.spin_until_future_complete(self, constitution_future)
        constitution_response = constitution_future.result()

        identity_request = Trigger.Request()

        identity_future = self.identity_service.call_async(identity_request)
        rclpy.spin_until_future_complete(self, identity_future)
        identity_response = identity_future.result()

        system_prompt = f"""
        Constitution:
        {constitution_response.message}

        Identity:
        {identity_response.message}

        You are a helpful assistant. You converse with users.
        Assume the conversation is carried over a voice interface, so try not to overwhelm the user.
        If you have multiple questions, please ask them one by one allowing user to respond before
        moving forward to the next question. Keep the conversation short and to the point.
        """

        self.history.append(SystemMessage(content=system_prompt))

        self.get_logger().info(f"System prompt initialized: {system_prompt}")

    def handle_human_message(self, human_ros_msg: String):
        if not human_ros_msg.data:
            self.get_logger().warn("Received an empty message, discarding")
            return

        self.history.append(HumanMessage(human_ros_msg.data))

        ai_msg: AIMessage = self.llm_with_tools.invoke(self.history)
        self.history.append(ai_msg)

        if ai_msg.content:
            self.send_message_to_human(ai_msg.content)

        # Claude 3.5 Sonnet on AWS Bedrock seems not to include any message content
        # if a tool is run. We need to to mitigate this issue by appending a tool
        # message after the AI message that requested tool use.
        for tool_call in ai_msg.tool_calls:
            task_json = tool_call["args"]["task"]
            task = Task.model_validate(task_json)
            self.add_task_to_queue(task)

            self.history.append(
                ToolMessage("Task added!", tool_call_id=tool_call["id"])
            )

        # Request a new AI message if any tools have been run.
        if ai_msg.tool_calls:
            new_ai_msg: AIMessage = self.llm_with_tools.invoke(self.history)
            self.history.append(new_ai_msg)

            if new_ai_msg.content:
                self.send_message_to_human(new_ai_msg.content)

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
