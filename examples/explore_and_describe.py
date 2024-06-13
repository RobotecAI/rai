from typing import Dict, List, Optional, Type, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage as _HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai.chat_models import ChatOpenAI

from rai.scenario_engine.tool_runner import run_requested_tools
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
    TF2TransformFetcher,
)


class HumanMessage(_HumanMessage):  # handle images
    def __init__(
        self,
        content: Union[str, List[Union[str, Dict]]],
        images: Optional[List[str]] = None,
    ):
        images = images or []
        final_content = [
            {"type": "text", "text": content},
        ]
        images_prepared = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image}",
                },
            }
            for image in images
        ]
        final_content.extend(images_prepared)
        super().__init__(content=final_content)


class DescribeAreaToolInput(BaseModel):
    """Input for the describe_area tool."""

    image_topic: str = Field(..., description="ROS2 image topic to subscribe to")


class DescribeAreaTool(BaseTool):
    """Describe the area"""

    name: str = "DescribeAreaTool"
    description: str = "A tool for describing the area around the robot."
    args_schema: Type[DescribeAreaToolInput] = DescribeAreaToolInput

    llm: BaseChatModel  # without tools
    system_message: SystemMessage

    def _run(self, image_topic: str):
        get_camera_image_tool = GetCameraImageTool()
        set_waypoint_tool = SetWaypointTool()

        current_position = TF2TransformFetcher().get_data()
        image = get_camera_image_tool.run(image_topic)["images"]
        llm_with_tools = self.llm.bind_tools([set_waypoint_tool])  # type: ignore
        human_message = HumanMessage(
            content=f"Describe the area around the robot (area, not items). Reason how would you name the room you are currently in"
            f". Use available tooling. Your current position is: {current_position}",
            images=image,
        )
        messages = [self.system_message, human_message]
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        run_requested_tools(ai_msg, [set_waypoint_tool], messages)
        return "Description of the area completed."


from typing import List

from langchain_aws.chat_models import ChatBedrock
from langchain_openai import ChatOpenAI

from rai.scenario_engine.messages import AgentLoop
from rai.scenario_engine.scenario_engine import ScenarioPartType, ScenarioRunner
from rai.tools.hmi_tools import WaitForSecondsTool
from rai.tools.ros.cat_demo_tools import FinishTool
from rai.tools.ros.tools import (
    GetCameraImageTool,
    GetCurrentPositionTool,
    SetWaypointTool,
)

DESCRIBER_PROMPT = """
You are an expert in describing the environment around you. Your main goal is to describe the area based on what you see in the image.
"""


def main():
    # simple_llm = ChatBedrock(
    #     model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-west-2"
    # )
    simple_llm = ChatOpenAI(model="gpt-4o")
    tools = [
        # GetCameraImageTool(),
        GetOccupancyGridTool(),
        # WaitForSecondsTool(),
        SetGoalPoseTool(),
        Ros2TopicTool(),
        DescribeAreaTool(
            llm=simple_llm, system_message=SystemMessage(content=DESCRIBER_PROMPT)
        ),
        FinishTool(),
    ]

    scenario: List[ScenarioPartType] = [
        SystemMessage(
            content="You are an autonomous agent. Your main goal is to fulfill the user's requests. "
            "Do not make assumptions about the environment you are currently in. "
            "Use the tooling provided to gather information about the environment."
            # "You are always required to send a voice message to the user about your decisions. This is crucial."
            # "The voice message should contain a very short information about what is going on and what is the next step. "
        ),
        HumanMessage(
            content="Describe your surroundings and gather more information as needed. "
            "Move to explore further, aiming for clear areas near the robot (red arrow). Make sure to describe the area during movement."
            "Utilize available methods to obtain the map and identify relevant data streams. "
            "Before starting the exploration, find out what kind of tooling is available and based on that plan your exploration."
        ),
        AgentLoop(stop_action=FinishTool().__class__.__name__, stop_iters=10),
    ]

    # llm = ChatBedrock(
    #     model_id="anthropic.claude-3-opus-20240229-v1:0", region_name="us-west-2"
    # )
    llm = ChatOpenAI(model="gpt-4o")
    runner = ScenarioRunner(scenario, llm, tools=tools)
    runner.run()
    runner.save_to_html()


if __name__ == "__main__":
    main()
