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

import os
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from rai.config.models import OPENAI_MULTIMODAL
from rai.scenario_engine.messages import AgentLoop
from rai.scenario_engine.scenario_engine import ScenarioPartType, ScenarioRunner
from rai.tools.hmi_tools import PlayVoiceMessageTool, WaitForSecondsTool
from rai.tools.ros.cat_demo_tools import FinishTool
from rai.tools.ros.cli import (
    Ros2InterfaceTool,
    Ros2ServiceTool,
    Ros2TopicTool,
    SetGoalPoseTool,
)
from rai.tools.ros.tools import (
    AddDescribedWaypointToDatabaseTool,
    GetCameraImageTool,
    GetCurrentPositionTool,
    GetOccupancyGridTool,
)


def main():
    tools: List[BaseTool] = [
        GetOccupancyGridTool(),
        GetCameraImageTool(),
        PlayVoiceMessageTool(),
        WaitForSecondsTool(),
        Ros2TopicTool(),
        Ros2ServiceTool(),
        Ros2InterfaceTool(),
        SetGoalPoseTool(),
        AddDescribedWaypointToDatabaseTool(),
        GetCurrentPositionTool(),
        FinishTool(),
    ]

    scenario: List[ScenarioPartType] = [
        SystemMessage(
            content="You are an autonomous agent. Your main goal is to fulfill the user's requests. "
            "Do not make assumptions about the environment you are currently in. "
            "Use the tooling provided to gather information about the environment."
            "You are always required to send a voice message to the user about your decisions. This is crucial."
            "The voice message should contain a very short information about what is going on and what is the next step. "
        ),
        HumanMessage(
            content="The robot is moving. Use vision to understand the surroundings, and add waypoints based on observations. camera is accesible at topic /camera/camera/color/image_raw ."
        ),
        AgentLoop(
            tools=tools, stop_tool=FinishTool().__class__.__name__, stop_iters=50
        ),
    ]
    log_usage = all((os.getenv("LANGFUSE_PK"), os.getenv("LANGFUSE_SK")))
    llm = ChatOpenAI(**OPENAI_MULTIMODAL)
    runner = ScenarioRunner(
        scenario,
        llm,
        llm_type="openai",
        scenario_name="Husarion example",
        log_usage=log_usage,
    )
    runner.run()


if __name__ == "__main__":
    main()
