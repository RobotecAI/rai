# Copyright (C) 2024 Robotec.AI
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
#
import base64
import io
import logging
import sys
from pprint import pformat
from typing import List, Optional, cast

import rclpy
import streamlit as st
from langchain.tools import tool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_openai.chat_models import ChatOpenAI
from PIL import Image
from pydantic import BaseModel
from streamlit.delta_generator import DeltaGenerator

from rai.agents.conversational_agent import create_conversational_agent
from rai.agents.state_based import get_stored_artifacts
from rai.messages import HumanMultimodalMessage
from rai_hmi.base import BaseHMINode
from rai_hmi.task import Task

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="LangChain Chat App", page_icon="🦜")

MODEL = "gpt-4o-mini"


# ---------- Cached Resources ----------
@st.cache_resource
def parse_args():
    robot_description_package = sys.argv[1] if len(sys.argv) > 1 else None
    return robot_description_package


@st.cache_resource
def initialize_agent(_node: BaseHMINode):
    llm = ChatOpenAI(
        temperature=0.5,
        model=MODEL,
        streaming=True,
    )

    @tool
    def add_task_to_queue(task: Task):
        """Use this tool to add a task to the queue. The task will be handled by the executor part of your system."""
        _node.add_task_to_queue(task)
        return f"Task added to the queue: {task.json()}"

    tools = [add_task_to_queue]
    agent = create_conversational_agent(
        llm, _node.tools + tools, _node.system_prompt, logger=_node.get_logger()
    )
    return agent


@st.cache_resource
def initialize_ros_node(robot_description_package: Optional[str]):
    rclpy.init()

    node = BaseHMINode(
        f"{robot_description_package}_hmi_node",
        robot_description_package=robot_description_package,
    )

    return node


# ---------- Helpers ----------
class EMOJIS:
    human = "🧑‍💻"
    bot = "🤖"
    tool = "🛠️"
    unknown = "❓"
    success = "✅"
    failure = "❌"


def display_agent_message(
    message: BaseMessage, tool_chat_obj: Optional[DeltaGenerator] = None
):
    """
    Display LLM message in streamlit UI.

    Args:
        message: The message to display
        tool_chat_obj: Pre-existing Streamlit object for the tool message.

    """
    message.content = cast(str, message.content)  # type: ignore
    if not message.content:
        return  # Tool messages might not have any content, skip displying them
    if isinstance(message, HumanMessage):
        if isinstance(
            message, HumanMultimodalMessage
        ):  # we do not handle user's images
            return
        st.chat_message("user", avatar=EMOJIS.human).markdown(message.content)
    elif isinstance(message, AIMessage):
        if message.content == "":
            return
        st.chat_message("bot", avatar=EMOJIS.bot).markdown(message.content)
    elif isinstance(message, ToolMessage):
        tool_call = st.session_state.tool_calls[message.tool_call_id]
        label = tool_call.name + " status: "
        status = EMOJIS.success if message.status == "success" else EMOJIS.failure
        if not tool_chat_obj:
            tool_chat_obj = st.expander(label=label + status).chat_message(
                "bot", avatar=EMOJIS.tool
            )
        with tool_chat_obj:
            st.markdown(message.content)
            artifacts = get_stored_artifacts(message.tool_call_id)
            for artifact in artifacts:
                if "images" in artifact:
                    base_64_image = artifact["images"][0]
                    image = decode_base64_into_image(base_64_image)
                    st.image(image)
    elif isinstance(message, SystemMessage):
        return  # we do not handle system messages
    else:
        raise ValueError("Unknown message type")


# ---------- Streamlit Application ----------
class SystemStatus(BaseModel):
    robot_database: bool
    system_prompt: bool


class Layout:
    """App has two columns: chat + robot state"""

    def __init__(self, robot_description_package) -> None:
        if robot_description_package:
            st.title(f"{robot_description_package.replace('_whoami', '')} chat app")
        else:
            st.title("ROS 2 Chat App")
            logging.warning(
                "No robot_description_package provided. Some functionalities may not work."
            )
        self.n_columns = 2
        self.tool_placeholders = dict()

    def draw(self, system_status: SystemStatus):
        self.draw_app_status_expander(system_status)
        self.draw_columns()

    def draw_app_status_expander(self, system_status: SystemStatus):
        with st.expander("System status", expanded=False):
            st.json(system_status.model_dump())

    def draw_columns(self):
        self.chat_column, self.mission_column = st.columns(self.n_columns)

    def create_tool_expanders(self, tool_calls: List[ToolCall]):
        with self.chat_column:
            for tool_call in tool_calls:
                tool_call = cast(ToolCall, tool_call)
                with st.expander(f"Tool call: {tool_call['name']}"):
                    st.markdown(f"Arguments: {tool_call['args']}")
                    self.tool_placeholders[tool_call["id"]] = st.empty()

    def write_chat_msg(self, msg: BaseMessage):
        with self.chat_column:
            if isinstance(msg, ToolMessage):
                display_agent_message(msg, self.tool_placeholders[msg.tool_call_id])
            else:
                display_agent_message(msg)

    def write_mission_msg(self, msg: BaseMessage):
        with self.mission_column:
            display_agent_message(msg)


class Memory:
    def __init__(self) -> None:
        if "mission_memory" not in st.session_state:
            st.session_state.mission_memory = []

        if "chat_memory" not in st.session_state:
            st.session_state.chat_memory = []

        if "tool_calls" not in st.session_state:
            st.session_state.tool_calls = {}

    @property
    def mission_memory(self):
        return st.session_state.mission_memory

    @property
    def chat_memory(self):
        return st.session_state.chat_memory

    @property
    def tool_calls(self):
        return st.session_state.tool_calls

    def register_tool_calls(self, tool_calls: List[ToolCall]):
        for tool_call in tool_calls:
            tool_call = cast(ToolCall, tool_call)
            tool_id = tool_call["id"]
            self.tool_calls[tool_id] = tool_call

    def __repr__(self) -> str:
        return f"===> Chat <===\n{pformat(self.chat_memory)}\n\n===> Mission <===\n{pformat(self.mission_memory)}\n\n===> Tool calls <===\n{pformat(self.tool_calls)}"


class Chat:
    def __init__(self, memory: Memory, layout: Layout) -> None:
        self.memory = memory
        self.layout = layout

    def user(self, txt):
        logger.info(f'User said: "{txt}"')
        msg = HumanMessage(content=txt)
        self.memory.chat_memory.append(msg)
        self.layout.write_chat_msg(msg)

    def bot(self, msg):
        logger.info(f'Bot said: "{msg}"')
        self.memory.chat_memory.append(msg)
        self.layout.write_chat_msg(msg)

    def tool(self, msg):
        logger.info(f'Tool said: "{msg}"')
        self.memory.chat_memory.append(msg)
        self.layout.write_chat_msg(msg)


class Agent:
    def __init__(self, node, memory) -> None:
        self.agent = initialize_agent(node)
        self.memory = memory

    def stream(self):
        # Copy, because agent's memory != streamlit app memory. App memory is used to
        # recreate the page, agent might manipulate it's history to perform the task.
        # In simplest case it adds thinker message to the state.
        messages = self.memory.chat_memory.copy()
        logger.info(f"Sending messages:\n{pformat(messages)}")

        return self.agent.stream({"messages": messages})


class StreamlitApp:
    def __init__(self, robot_description_package) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.robot_description_package = robot_description_package

        self.layout = Layout(self.robot_description_package)
        self.memory = Memory()
        self.chat = Chat(self.memory, self.layout)

        self.node = self.initialize_node(robot_description_package)
        self.agent = Agent(self.node, self.memory)

    def run(self):
        system_status = self.get_system_status()
        self.layout.draw(system_status)

        self.recreate_from_memory()
        self.interact()

    def get_system_status(self) -> SystemStatus:
        return SystemStatus(
            robot_database=self.node.faiss_index is not None,
            system_prompt=self.node.system_prompt == "",
        )

    def initialize_node(self, robot_description_package):
        with st.spinner("Initializing ROS 2 node..."):
            node = initialize_ros_node(robot_description_package)
            self.logger.info("ROS 2 node initialized")
        return node

    def recreate_from_memory(self, max_display: int = 5):
        """
        Recreates the page from the memory. It is because Streamlit reloads entire page every widget action.
        See: https://docs.streamlit.io/get-started/fundamentals/main-concepts

        Args:
            max_display (int, optional): Max number of messages to display. Rest will be hidden in a toggle. Defaults to 5.

        """
        with self.layout.chat_column:
            if len(self.memory.chat_memory) > max_display:
                n_hide = len(self.memory.chat_memory) - max_display
                hidden_history = self.memory.chat_memory[:n_hide]
                with st.expander("Untoggle to see full chat history"):
                    for message in hidden_history:
                        display_agent_message(message)
                for message in self.memory.chat_memory[-max_display:]:
                    display_agent_message(message)
            else:
                for message in self.memory.chat_memory:
                    display_agent_message(message)

        with self.layout.mission_column:
            for message in self.memory.mission_memory:
                display_agent_message(message)

    def interact(self):
        prompt = st.chat_input("What is your question?")
        if not prompt:
            return

        self.chat.user(prompt)

        message_placeholder = st.container()
        with message_placeholder, st.spinner("Thinking..."):
            for state in self.agent.stream():
                logger.info(f"State:\n{pformat(state)}")
                node_name = list(state.keys())[0]
                if node_name == "thinker":
                    last_message = state[node_name]["messages"][-1]
                    self.chat.bot(last_message)
                    self.memory.register_tool_calls(last_message.tool_calls)
                    self.layout.create_tool_expanders(last_message.tool_calls)

                elif node_name == "tools":
                    last_ai_msg_idx = 0
                    for message in state[node_name]["messages"]:
                        if isinstance(message, AIMessage):
                            last_ai_msg_idx = state[node_name]["messages"].index(
                                message
                            )

                    for message in state[node_name]["messages"][
                        last_ai_msg_idx + 1 :  # noqa: E203
                    ]:
                        if message.type == "tool":
                            self.memory.tool_calls[message.tool_call_id] = message
                            self.chat.tool(message)
                        else:
                            break


def decode_base64_into_image(base64_image: str):
    image = Image.open(io.BytesIO(base64.b64decode(base64_image)))
    return image


if __name__ == "__main__":
    robot_description_package = parse_args()
    app = StreamlitApp(robot_description_package)
    app.run()
