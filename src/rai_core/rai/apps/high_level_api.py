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


from typing import List

from langchain_core.messages import BaseMessage, HumanMessage

from rai.agents.conversational_agent import create_conversational_agent
from rai.tools.ros2.cli import ros2_action, ros2_interface, ros2_service, ros2_topic
from rai.utils.model_initialization import get_llm_model


class Agent:
    def __init__(self):
        self.history: List[BaseMessage] = []
        self.llm = get_llm_model(model_type="complex_model")


class ROS2Agent(Agent):
    def __init__(self):
        super().__init__()
        self.tools = [
            ros2_topic,
            ros2_interface,
            ros2_service,
            ros2_action,
        ]
        self.agent = create_conversational_agent(
            self.llm, self.tools, "You are a ROS2 expert."
        )

    def __call__(self, message: str):
        self.history.append(HumanMessage(content=message))
        response = self.agent.invoke({"messages": self.history})
        output = response["messages"][-1].content
        return output
