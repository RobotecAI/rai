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

import logging
import os
from typing import List

import rclpy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from rai.config.models import OPENAI_MULTIMODAL
from rai.node import RaiNode
from rai.scenario_engine.messages import AgentLoop
from rai.scenario_engine.scenario_engine import ScenarioPartType, ScenarioRunner
from rai.tools.ros.native import Ros2ShowMsgInterfaceTool
from rai.tools.ros.native_actions import (
    Ros2ActionRunner,
    Ros2CancelAction,
    Ros2CheckActionResults,
    Ros2GetActionNamesAndTypesTool,
    Ros2GetRegisteredActions,
)
from rai.tools.ros.tools import GetCurrentPositionTool, GetOccupancyGridTool
from rai.tools.time import sleep_max_5s


def main():

    log_usage = all((os.getenv("LANGFUSE_PK"), os.getenv("LANGFUSE_SK")))
    llm = ChatOpenAI(**OPENAI_MULTIMODAL)

    rclpy.init()

    rai_node = RaiNode()  # type: ignore

    tools: List[BaseTool] = [
        Ros2GetActionNamesAndTypesTool(node=rai_node),
        # Ros2GetTopicsNamesAndTypesTool(node=rai_node),
        # Ros2PubMessageTool(node=rai_node),
        Ros2ShowMsgInterfaceTool(),
        Ros2ActionRunner(node=rai_node),
        Ros2CheckActionResults(node=rai_node),
        Ros2CancelAction(node=rai_node),
        # Ros2ListActionFeedbacks(node=rai_node),
        Ros2GetRegisteredActions(node=rai_node),
        GetCurrentPositionTool(),
        GetOccupancyGridTool(),
        sleep_max_5s,
    ]

    scenario: List[ScenarioPartType] = [
        SystemMessage(
            content="You are an autonomous agent. Your main goal is to fulfill the user's requests. "
            "Do not make assumptions about the environment you are currently in. "
            "Use the tooling provided to gather information about the environment."
        ),
        HumanMessage(content="Drive around close to the wall"),
        AgentLoop(tools=tools, stop_tool=None, stop_iters=50),
    ]

    runner = ScenarioRunner(
        scenario,
        llm,
        ros_node=rai_node,
        llm_type="openai",
        scenario_name="Nav2 example",
        log_usage=log_usage,
        logging_level=logging.INFO,
    )

    runner.run()
    rai_node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
