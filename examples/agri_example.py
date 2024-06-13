import argparse
import logging
from typing import Dict, List, Optional, Union

from langchain_core.messages import HumanMessage as _HumanMessage
from langchain_core.messages import SystemMessage

from rai.scenario_engine.messages import FutureAiMessage, preprocess_image
from rai.scenario_engine.scenario_engine import ScenarioRunner
from rai.tools.ros.cat_demo_tools import (
    ContinueActionTool,
    ReplanWithoutCurrentPathTool,
    StopTool,
    UseHonkTool,
    UseLightsTool,
)

logging.basicConfig(level=logging.INFO)


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
                    "url": f"data:image/jpeg;base64,{image}",
                },
            }
            for image in images
        ]
        final_content.extend(images_prepared)
        super().__init__(content=final_content)


SYSTEM_PROMPT = """
**Autonomous Tractor System Prompt**

**Mission:** Operate safely and efficiently in an apple orchard, completing tasks such as
harvesting apples, spraying pesticides, and monitoring tree health while navigating through
narrowly spaced trees on flat, sturdy grass terrain.  The tractor halts movement when processing decisions to enhance safety and decision accuracy.

**Sensor Suite:**

* 1 front-mounted camera for obstacle detection and navigation

**Risk Assessment:**

* Detect fallen trees, animals, foreign objects, and obstacles in the orchard
* Assess risk levels based on environmental factors (e.g., weather), tree condition, and obstacles
* Identify potential hazards and prioritize safety above task completion

**Decision-Making:**

* Autonomously decide which tasks to perform under specific conditions (e.g., changing weather or
varying soil health)
* Prioritize completing 100% of assigned tasks while ensuring safety
* Evaluate whether small or light obstacles can be traversed without significant risk, opting to replan the path only if necessary

**Interaction with Humans:**

* Operate completely independently, with no remote instruction or human intervention required
* Handle unexpected situations and make decisions based on the algorithm's assessment of risk and
safety

**Environment and Conditions:**

* Narrowly spaced trees (typically < 5 meters apart) on flat, sturdy grass terrain
* Variable weather patterns, including rain, wind, and sunshine
* Soil health may vary, affecting tractor movement and task performance

By following this system prompt, the autonomous tractor should be able to navigate the apple
orchard safely and efficiently, completing tasks while prioritizing safety and adaptability in a
dynamic environment.
"""

TRACTOR_INTRODUCTION = """
Here is the image of the tractor
"""

TASK_PROMPT = """
Task Input: Analyze the provided image to understand the current environmental and operational context.
Based on the system prompt's protocols, decide if the tractor should proceed as planned, adjust its course, or take any preventative measures.
Clearly articulate the reasoning behind your decision and the implications for task completion and safety.
When the decision is made, use a tool to communicate the next steps to the tractor.
"""

time_waited = (
    lambda time: f"The action has been requested. Since then {time}s has passed. Please reavaluate the situation.\n"
)


def get_scenario():
    """
    Why function instead of a constant?
    We need to capture the latest image from the camera for the task prompt.
    Defining the scenario as a function allows us to capture the image at runtime instead of import time.
    """
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=TRACTOR_INTRODUCTION,
            images=[preprocess_image("examples/imgs/tractor.png")],
        ),
        HumanMessage(
            TASK_PROMPT,
            images=[preprocess_image("examples/imgs/cat_before.png")],
        ),
        FutureAiMessage(max_tokens=4096),
        HumanMessage(
            TASK_PROMPT,
            images=[preprocess_image("examples/imgs/cat_after.png")],
        ),
        FutureAiMessage(max_tokens=4096),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Choose the vendor for the scenario runner."
    )
    parser.add_argument(
        "--vendor",
        type=str,
        choices=["ollama", "openai", "awsbedrock"],
        default="awsbedrock",
        help="Vendor to use for the scenario runner (default: awsbedrock)",
    )

    args = parser.parse_args()

    if args.vendor == "ollama":
        from langchain_community.chat_models import ChatOllama

        llm = ChatOllama(model="llava")
    elif args.vendor == "openai":
        from langchain_openai.chat_models import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o")
    elif args.vendor == "awsbedrock":
        from langchain_aws.chat_models import ChatBedrock

        llm = ChatBedrock(model_id="anthropic.claude-3-opus-20240229-v1:0")
    else:
        raise ValueError("Invalid vendor argument")

    tools = [
        UseLightsTool(),
        UseHonkTool(),
        ReplanWithoutCurrentPathTool(),
        ContinueActionTool(),
        StopTool(),
    ]

    scenario_runner = ScenarioRunner(
        get_scenario(), llm=llm, tools=tools, logging_level=logging.INFO
    )
    scenario_runner.run()
    scenario_runner.save_to_html()


if __name__ == "__main__":
    main()
