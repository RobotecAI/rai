import logging
import os
from typing import List

import rclpy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from rclpy.node import Node

from rai.config.models import OPENAI_MULTIMODAL
from rai.scenario_engine.messages import AgentLoop
from rai.scenario_engine.scenario_engine import ScenarioPartType, ScenarioRunner
from rai.tools.ros.cat_demo_tools import FinishTool
from rai.tools.ros.native import (
    Ros2GetOneMsgFromTopicTool,
    Ros2GetTopicsNamesAndTypesTool,
    Ros2PubMessageTool,
    Ros2ShowRos2MsgInterfaceTool,
)


def main():
    log_usage = all((os.getenv("LANGFUSE_PK"), os.getenv("LANGFUSE_SK")))
    llm = ChatOpenAI(**OPENAI_MULTIMODAL)

    rclpy.init()

    rai_node = Node("rai")  # type: ignore

    tools: List[BaseTool] = [
        Ros2GetTopicsNamesAndTypesTool(),
        Ros2GetOneMsgFromTopicTool(node=rai_node),
        Ros2PubMessageTool(node=rai_node),
        Ros2ShowRos2MsgInterfaceTool(),
        FinishTool(),
    ]

    scenario: List[ScenarioPartType] = [
        SystemMessage(
            content="You are an autonomous agent. Your main goal is to fulfill the user's requests. "
            "Do not make assumptions about the environment you are currently in. "
            "Use the tooling provided to gather information about the environment."
        ),
        HumanMessage(content="The robot is moving. Send robot to the random location"),
        AgentLoop(
            tools=tools, stop_tool=FinishTool().__class__.__name__, stop_iters=50
        ),
    ]

    runner = ScenarioRunner(
        scenario,
        llm,
        llm_type="openai",
        scenario_name="Husarion example",
        logging_level=logging.INFO,
        log_usage=log_usage,
        logging_level=logging.INFO,
    )

    runner.run()
    rai_node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
