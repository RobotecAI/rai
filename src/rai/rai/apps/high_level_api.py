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

from typing import List, Literal

from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from rai.agents.conversational_agent import create_conversational_agent
from rai.config.models import BEDROCK_MULTIMODAL, OPENAI_MULTIMODAL
from rai.tools.ros.cli import (
    Ros2ActionTool,
    Ros2InterfaceTool,
    Ros2ServiceTool,
    Ros2TopicTool,
)


class Agent:
    def __init__(self, vendor: Literal["openai", "bedrock"]):
        self.vendor = vendor
        self.history: List[BaseMessage] = []
        if vendor == "openai":
            self.llm = ChatOpenAI(**OPENAI_MULTIMODAL)
        else:
            self.llm = ChatBedrock(**BEDROCK_MULTIMODAL)


class ROS2Agent(Agent):
    def __init__(self, vendor: Literal["openai", "bedrock"]):
        super().__init__(vendor)
        self.tools = [
            Ros2TopicTool(),
            Ros2InterfaceTool(),
            Ros2ServiceTool(),
            Ros2ActionTool(),
        ]
        self.agent = create_conversational_agent(
            self.llm, self.tools, "You are a ROS2 expert."
        )

    def __call__(self, message: str):
        self.history.append(HumanMessage(content=message))
        response = self.agent.invoke({"messages": self.history})
        output = response["messages"][-1].content
        return output
