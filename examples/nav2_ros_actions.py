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
from rai.tools.ros.native import (
    Ros2ActionRunner,
    Ros2CancelAction,
    Ros2CheckActionResults,
    Ros2GetActionNamesAndTypesTool,
    Ros2GetRegisteredActions,
    Ros2ShowRos2MsgInterfaceTool,
)
from rai.tools.ros.tools import GetCurrentPositionTool, GetOccupancyGridTool
from rai.tools.time import sleep


def main():

    log_usage = all((os.getenv("LANGFUSE_PK"), os.getenv("LANGFUSE_SK")))
    llm = ChatOpenAI(**OPENAI_MULTIMODAL)

    rclpy.init()

    rai_node = RaiNode()  # type: ignore

    tools: List[BaseTool] = [
        # Ros2GetTopicsNamesAndTypesTool(),
        # Ros2GetOneMsgFromTopicTool(node=rai_node),
        # Ros2PubMessageTool(node=rai_node),
        Ros2ActionRunner(node=rai_node),
        Ros2CheckActionResults(node=rai_node),
        Ros2GetActionNamesAndTypesTool(node=rai_node),
        Ros2ShowRos2MsgInterfaceTool(node=rai_node),
        GetOccupancyGridTool(),
        GetCurrentPositionTool(),
        Ros2GetRegisteredActions(node=rai_node),
        Ros2CancelAction(node=rai_node),
        sleep,
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
        scenario_name="Husarion example",
        log_usage=log_usage,
        logging_level=logging.INFO,
    )

    runner.run()
    rai_node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
