import base64
import sys
from io import BytesIO

import rclpy
import streamlit as st
from ament_index_python.packages import get_package_share_directory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_community.callbacks.streamlit.streamlit_callback_handler import (
    StreamlitCallbackHandler,
)
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from PIL import Image
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from rai.scenario_engine.messages import HumanMultimodalMessage
from rai.tools.ros.native import Ros2GetTopicsNamesAndTypesTool
from rai.tools.ros.tools import GetCameraImageTool
from rai_hmi.task import Task
from rai_interfaces.srv import VectorStoreRetrieval

st.set_page_config(page_title="LangChain Chat App", page_icon="🦜")
st.title("LangChain Chat App")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )


@st.cache_resource
def initialize_ros(robot_description_package: str):

    rclpy.init()

    class HMINode(Node):
        def __init__(self, robot_description_package: str):
            super().__init__("rai_hmi_node")
            self.callback_group = ReentrantCallbackGroup()
            self.robot_description_package = robot_description_package
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

        def initialize_system_prompt(self):
            while not self.constitution_service.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(
                    "Constitution service not available, waiting again..."
                )

            while not self.identity_service.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(
                    "Identity service not available, waiting again..."
                )

            constitution_request = Trigger.Request()

            constitution_future = self.constitution_service.call_async(
                constitution_request
            )
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
            Always reply in first person. When you use the tool and get the output, always present it in first person.
            """

            self.get_logger().info(f"System prompt initialized: {system_prompt}")
            return system_prompt

        def load_documentation(self) -> FAISS:
            faiss_index = FAISS.load_local(
                get_package_share_directory(self.robot_description_package)
                + "/description",
                OpenAIEmbeddings(),
                allow_dangerous_deserialization=True,
            )
            return faiss_index

    hmi_node = HMINode(robot_description_package=robot_description_package)
    system_prompt = hmi_node.initialize_system_prompt()
    faiss_index = hmi_node.load_documentation()
    return hmi_node, system_prompt, faiss_index


llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
    streaming=True,
)


@tool
def add_task_to_queue(task: Task):
    """Use this tool to add a task to the queue. The task will be handled by the executor part of your system."""
    hmi_node.task_addition_request_publisher.publish(String(data=task.json()))
    return f"Task added to the queue: {task.json()}"


@tool
def get_image_from_topic(topic: str):
    """Use this tool to get an image from a ROS 2 topic."""
    tool = GetCameraImageTool()
    output = tool._run(topic=topic)
    msg = HumanMultimodalMessage(
        content="Please describe the image.", images=output["images"]
    )
    ai_msg = llm.invoke([msg])
    image_data = base64.b64decode(output["images"][0])
    image_buffer = BytesIO(image_data)
    pil_image = Image.open(image_buffer)
    st.container(border=True).image(pil_image, use_column_width=True)
    return f"Image description: {ai_msg.content}"


@tool
def search_database(query: str) -> str:
    """Use this tool to search the documentation database."""
    results = faiss_index.similarity_search_with_score(query)
    formatted_results = []
    for idx, (document, score) in enumerate(results):
        formatted_results.append(
            f"Result {idx + 1}:\nDocument: {document}\nScore: {score:.4f}\n"
        )
    return "\n".join(formatted_results)


@st.cache_resource
def initialize_genAI(system_prompt: str, _node: Node):
    search = DuckDuckGoSearchRun()
    tools = [
        search,
        Ros2GetTopicsNamesAndTypesTool(node=_node),
        get_image_from_topic,
        add_task_to_queue,
        search_database,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            ("human", "Chat History: {chat_history}"),
            ("human", "Agent Scratchpad: {agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        memory=st.session_state.memory,
    )

    return agent_executor


hmi_node, system_prompt, faiss_index = initialize_ros(sys.argv[1])
agent_executor = initialize_genAI(system_prompt=system_prompt, _node=hmi_node)

st.subheader("Chat")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.chat_message("user").markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        cfg = RunnableConfig(callbacks=[cb])
        message_placeholder = st.empty()
        full_response = ""

        response = agent_executor.invoke({"input": prompt}, config=cfg)
        if "output" in response:
            full_response = response["output"]
        else:
            full_response = str(response)

        message_placeholder.markdown(full_response)

    st.session_state["messages"].append({"role": "assistant", "content": full_response})
