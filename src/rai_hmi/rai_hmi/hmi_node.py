#  Copyright 2024 Robotec.ai
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import List

import rclpy
from ament_index_python.packages import get_package_share_directory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
from std_srvs.srv import Trigger

from rai_interfaces.srv import VectorStoreRetrieval

from .task import Task
from .tools import QueryDatabaseTool, QueueTaskTool


class HMINode(Node):
    def __init__(self):
        super().__init__("rai_hmi_node")
        self.declare_parameter("robot_description_package", Parameter.Type.STRING)
        # TODO: add parameter to choose model

        self.history: List[BaseMessage] = []

        self.callback_group = ReentrantCallbackGroup()
        self.hmi_subscription = self.create_subscription(
            String,
            "from_human",
            self.handle_human_message,
            10,
            callback_group=self.callback_group,
        )
        self.processing = False
        self.hmi_publisher = self.create_publisher(
            String, "to_human", 10, callback_group=self.callback_group
        )

        self.create_timer(
            0.01, self.status_callback, callback_group=self.callback_group
        )

        self.status_publisher = self.create_publisher(String, "hmi_status", 10)  # type: ignore
        self.task_addition_request_publisher = self.create_publisher(
            String, "task_addition_requests", 10
        )

        self.documentation_service = self.create_client(
            VectorStoreRetrieval,
            "rai_whoami_documentation_service",
            callback_group=self.callback_group,
        )
        self.constitution_service = self.create_client(
            Trigger,
            "rai_whoami_constitution_service",
        )
        self.identity_service = self.create_client(
            Trigger, "rai_whoami_identity_service"
        )

        self.robot_description_package = (
            self.get_parameter("robot_description_package")
            .get_parameter_value()
            .string_value
        )  # type: ignore

        self.get_logger().info("HMI Node has been started")
        system_prompt = self.initialize_system_prompt()

        llm = ChatOpenAI(model="gpt-4o")

        tools = [
            QueryDatabaseTool(get_response=self.get_database_response),
            QueueTaskTool(add_task=self.add_task_to_queue),
        ]
        self.name_to_tool_map = {tool.name: tool for tool in tools}

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        self.agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)
        self.faiss_index = self._load_documentation()

    def status_callback(self):
        if self.processing:
            self.status_publisher.publish(String(data="processing"))
        else:
            self.status_publisher.publish(String(data="waiting"))

    def get_database_response(self, query: str) -> VectorStoreRetrieval.Response:
        # The following code is a 1:1 replacement for the commented out code below
        # The same code is used in the rai_whoami_node.py for the service callback
        response = VectorStoreRetrieval.Response()
        output = self.faiss_index.similarity_search_with_score(query)
        response.message = "Query successful"
        response.success = True
        response.documents = [doc.page_content for doc, _ in output]
        response.scores = [float(score) for _, score in output]
        return response

        # Running the following code is problematic because
        # it works once, but then it blocks the handle_human_message
        while not self.documentation_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Identity service not available, waiting again...")

        request = VectorStoreRetrieval.Request()
        request.query = query

        request_future = self.documentation_service.call_async(request)
        rclpy.spin_until_future_complete(self, request_future)
        request_result = request_future.result()
        return request_result

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

        self.get_logger().info(f"System prompt initialized: {system_prompt}")
        return system_prompt

    def handle_human_message(self, human_ros_msg: String):
        self.processing = True
        self.get_logger().info("Handling human message")
        if not human_ros_msg.data:
            self.get_logger().warn("Received an empty message, discarding")
            self.processing = False
            return
        response = self.agent_executor.invoke(
            {"input": human_ros_msg.data, "chat_history": self.history}
        )
        self.history.append(HumanMessage(human_ros_msg.data))
        self.history.append(SystemMessage(content=response["output"]))
        self.send_message_to_human(response["output"])
        self.get_logger().info("Finished handling human message")
        self.processing = False

    def _load_documentation(self) -> FAISS:
        faiss_index = FAISS.load_local(
            get_package_share_directory(self.robot_description_package)
            + "/description",
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True,
        )
        return faiss_index

    def send_message_to_human(self, content: str):
        msg = String()
        msg.data = content
        self.hmi_publisher.publish(msg)

    def add_task_to_queue(self, task: Task):
        msg = String()
        msg.data = task.json()
        self.task_addition_request_publisher.publish(msg)

    # TODO: This method is currently not called anywhere
    def clear_history(self):
        # Remove everything except system prompt
        self.history = self.history[:1]


def main():
    rclpy.init()
    hmi_node = HMINode()
    executor = MultiThreadedExecutor()
    executor.add_node(hmi_node)
    executor.spin()
    hmi_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
