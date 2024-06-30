import os
from typing import List

import rclpy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rai.scenario_engine.messages import AgentLoop
from rai.scenario_engine.scenario_engine import ScenarioPartType, ScenarioRunner
from rai.tools.ros.cat_demo_tools import FinishTool
from rai.tools.ros.native import (
    Ros2GetOneMsgFromTopicTool,
    Ros2PubMessageTool,
    Ros2GetTopicsNamesAndTypesTool
)

from rclpy.node import Node
from ros2cli.node.strategy import NodeStrategy


def main():
    scenario: List[ScenarioPartType] = [
        SystemMessage(
            content="You are an autonomous agent. Your main goal is to fulfill the user's requests. "
            "Do not make assumptions about the environment you are currently in. "
            "Use the tooling provided to gather information about the environment."
        ),
        HumanMessage(content="The robot is moving. Send robot to the random location"),
        AgentLoop(stop_action=FinishTool().__class__.__name__, stop_iters=50),
    ]

    log_usage = all((os.getenv("LANGFUSE_PK"), os.getenv("LANGFUSE_SK")))
    llm = ChatOpenAI(model="gpt-4o")

    rclpy.init()

    rai_node = Node("rai") # type: ignore


    runner = ScenarioRunner( 
        scenario,
        llm,
        llm_type="openai",
        scenario_name="Husarion example",
        log_usage=log_usage,
        logging_level="DEBUG"
    )

    tools = [
        Ros2GetTopicsNamesAndTypesTool(),
        Ros2GetOneMsgFromTopicTool(node=rai_node),
        Ros2PubMessageTool(node=rai_node),
        FinishTool(),
    ]

    runner.bind_tools(tools)

    runner.run()
    rai_node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
