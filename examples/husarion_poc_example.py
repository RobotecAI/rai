from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rai.scenario_engine.messages import AgentLoop
from rai.scenario_engine.scenario_engine import ScenarioPartType, ScenarioRunner
from rai.tools.hmi_tools import PlayVoiceMessageTool, WaitForSecondsTool
from rai.tools.ros.cat_demo_tools import FinishTool
from rai.tools.ros.cli_tools import (
    Ros2InterfaceTool,
    Ros2ServiceTool,
    Ros2TopicTool,
    SetGoalPoseTool,
)
from rai.tools.ros.tools import (
    GetCameraImageTool,
    GetCurrentPositionTool,
    GetOccupancyGridTool,
    SetWaypointTool,
)


def main():
    tools = [
        GetOccupancyGridTool(),
        GetCameraImageTool(),
        PlayVoiceMessageTool(),
        WaitForSecondsTool(),
        Ros2TopicTool(),
        Ros2ServiceTool(),
        Ros2InterfaceTool(),
        SetGoalPoseTool(),
        SetWaypointTool(),
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
        AgentLoop(stop_action=FinishTool().__class__.__name__, stop_iters=50),
    ]

    llm = ChatOpenAI(model="gpt-4o")
    runner = ScenarioRunner(scenario, llm, tools=tools)
    runner.run()
    runner.save_to_html()


if __name__ == "__main__":
    main()
