import json
import os
from typing import List, Type

from langchain_aws import ChatBedrock
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from rai.communication.ros_communication import TF2TransformFetcher
from rai.config.models import BEDROCK_CLAUDE_HAIKU, BEDROCK_CLAUDE_SONNET
from rai.scenario_engine.messages import AgentLoop, HumanMultimodalMessage
from rai.scenario_engine.scenario_engine import ScenarioPartType, ScenarioRunner
from rai.scenario_engine.tool_runner import run_requested_tools
from rai.tools.ros.cat_demo_tools import FinishTool
from rai.tools.ros.cli import Ros2TopicTool, SetGoalPoseTool
from rai.tools.ros.tools import (
    AddDescribedWaypointToDatabaseTool,
    GetCameraImageTool,
    GetOccupancyGridTool,
)


class DescribeAreaToolInput(BaseModel):
    """Input for the describe_area tool."""

    image_topic: str = Field(..., description="ROS2 image topic to subscribe to")


class DescribeAreaTool(BaseTool):
    """
    Describe the area. The tool uses the available tooling to describe the area around the robot.
    The output is saved to the map database.
    The tool does not return anything specific to the tool run.
    """

    name: str = "DescribeAreaTool"
    description: str = "A tool for describing the area around the robot."
    args_schema: Type[DescribeAreaToolInput] = DescribeAreaToolInput

    llm: BaseChatModel  # without tools
    system_message: SystemMessage
    map_database: str = ""

    def _run(self, image_topic: str):
        get_camera_image_tool = GetCameraImageTool()
        set_waypoint_tool = AddDescribedWaypointToDatabaseTool(
            map_database=self.map_database
        )

        current_position = TF2TransformFetcher().get_data()
        image = get_camera_image_tool.run(image_topic)["images"]
        llm_with_tools = self.llm.bind_tools([set_waypoint_tool])  # type: ignore
        human_message = HumanMultimodalMessage(
            content=f"Describe the area around the robot (area, not items). Reason how would you name the room you are currently in"
            f". Use available tooling. Your current position is: {current_position}",
            images=image,
        )
        messages = [self.system_message, human_message]
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        run_requested_tools(
            ai_msg, [set_waypoint_tool], messages, llm_type="bedrock"
        )  # TODO(@maciejmajek): fix hardcoded llm_type
        return "Description of the area completed."


DESCRIBER_PROMPT = """
You are an expert in describing the environment around you. Your main goal is to describe the area based on what you see in the image.
"""


def main():
    # setup database for the example
    if not os.path.exists("map_database.json"):
        with open("map_database.json", "w") as f:
            json.dump([], f)

    simple_llm = ChatBedrock(**BEDROCK_CLAUDE_HAIKU)  # type: ignore[arg-missing]
    tools: List[BaseTool] = [
        GetOccupancyGridTool(),
        SetGoalPoseTool(),
        Ros2TopicTool(),
        DescribeAreaTool(
            llm=simple_llm,
            system_message=SystemMessage(content=DESCRIBER_PROMPT),
            map_database="map_database.txt",
        ),
        FinishTool(),
    ]

    scenario: List[ScenarioPartType] = [
        SystemMessage(
            content="You are an autonomous agent. Your main goal is to fulfill the user's requests. "
            "Do not make assumptions about the environment you are currently in. "
            "Use the tooling provided to gather information about the environment. Remember to list available topics. "
        ),
        HumanMultimodalMessage(
            content="Describe your surroundings and gather more information as needed. "
            "Move to explore further, aiming for clear areas near the robot (red arrow). Make sure to describe the area during movement."
            "Utilize available methods to obtain the map and identify relevant data streams. "
            "Before starting the exploration, find out what kind of tooling is available and based on that plan your exploration. For description, use the available tooling."
        ),
        AgentLoop(
            tools=tools, stop_tool=FinishTool().__class__.__name__, stop_iters=50
        ),
    ]

    llm = ChatBedrock(**BEDROCK_CLAUDE_SONNET)  # type: ignore[arg-missing]
    runner = ScenarioRunner(scenario, llm=llm, tools=tools, llm_type="bedrock")
    runner.run()
    runner.save_to_html()


if __name__ == "__main__":
    main()
