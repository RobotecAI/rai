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
from rai.tools.ros.cat_demo_tools import FinishTool
from rai.tools.ros.native import (
    Ros2ActionRunner,
    Ros2CheckActionResults,
    Ros2GetActionNamesAndTypesTool,
    Ros2ShowRos2MsgInterfaceTool,
)
from rai.tools.time import sleep


def main():

    log_usage = all((os.getenv("LANGFUSE_PK"), os.getenv("LANGFUSE_SK")))
    llm = ChatOpenAI(**OPENAI_MULTIMODAL)

    rclpy.init()
    rai_node = RaiNode()

    tools: List[BaseTool] = [
        Ros2GetActionNamesAndTypesTool(node=rai_node),
        Ros2ShowRos2MsgInterfaceTool(node=rai_node),
        Ros2ActionRunner(node=rai_node),
        Ros2CheckActionResults(node=rai_node),
        sleep,
        FinishTool(),
    ]

    scenario: List[ScenarioPartType] = [
        SystemMessage(
            content="You are an ros2 autonomous agent. Your main goal is to fulfill the user's requests using available ros2 interfaces."
        ),
        HumanMessage(
            content="Please calculate 2 fibbonacci sequences of 2 different random numbers from 1 to 10."
        ),
        AgentLoop(
            tools=tools, stop_tool=FinishTool().__class__.__name__, stop_iters=50
        ),
    ]

    runner = ScenarioRunner(
        scenario,
        llm,
        ros_node=rai_node,
        llm_type="openai",
        scenario_name="Simple example",
        log_usage=log_usage,
        logging_level=logging.INFO,
    )

    runner.run()
    rai_node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()