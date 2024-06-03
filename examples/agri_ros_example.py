import argparse
import logging

import rai.actions
from rai.actions.executor import ConditionalExecutor
from rai.communication.ros_communication import SingleImageGrabber
from rai.message import AssistantMessage, ConditionalMessage, ConstantMessage, Message
from rai.requirements import MessageLengthRequirement, RequirementSeverity
from rai.scenario_engine.scenario_engine import ScenarioRunner
from rai.vendors.vendors import AWSBedrockVendor, OllamaVendor, OpenAIVendor

logging.basicConfig(level=logging.INFO)


from nav_msgs.msg import Odometry

from rai.actions.actions import CallCommand, EventReportSaver, Wait
from rai.actions.executor import Executor
from rai.communication.ros_communication import SingleImageGrabber, SingleMessageGrabber
from rai.message import AssistantMessage, Message, SystemMessage, UserMessage
from rai.scenario_engine.scenario_engine import ConditionalScenario

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
"""

POSSIBLE_ACTIONS = """
Given your assessment of the risks, which action from this list should be executed?
use_lights
use_honk
replan_without_current_path
continue
Respond with only the action's name. Do not add any extra characters.
"""

action_to_command = {
    "use_lights": "ros2 topic pub --once /alert/flash std_msgs/Bool \"data: 'true'\"",
    "use_honk": "ros2 topic pub --once /alert/flash std_msgs/Bool \"data: 'true'\"",
    "replan_without_current_path": "ros2 topic pub /control/move std_msgs/msg/String \"{{data: 'REPLAN'}}\" --once",
    "continue": "ros2 topic pub /control/move std_msgs/msg/String \"{{data: 'CONTINUE'}}\" --once",
    "stop": "ros2 topic pub /control/move std_msgs/msg/String \"{{data: 'STOP'}}\" --once",
}

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
        SystemMessage(SYSTEM_PROMPT),
        UserMessage(
            TRACTOR_INTRODUCTION,
            images=[Message.preprocess_image("examples/imgs/tractor.png")],
        ),
        AssistantMessage(max_tokens=4096),
        UserMessage(
            TASK_PROMPT,
            images=[SingleImageGrabber(topic=f"/camera_image_color").get_data()],
        ),
        AssistantMessage(max_tokens=4096),
        UserMessage(POSSIBLE_ACTIONS),
        AssistantMessage(max_tokens=50),
        ConditionalScenario(
            if_true=[
                Executor(
                    CallCommand(
                        action_to_command=action_to_command,
                        separate_thread=False,
                    )
                ),
                Executor(Wait(seconds=5)),
                UserMessage(
                    time_waited(5) + TASK_PROMPT,
                    images=[SingleImageGrabber(topic=f"/camera_image_color").get_data],
                ),
                AssistantMessage(max_tokens=4096),
                UserMessage(POSSIBLE_ACTIONS),
                AssistantMessage(max_tokens=50),
                Executor(CallCommand(action_to_command=action_to_command)),
            ],
            if_false=[
                Executor(CallCommand(action_to_command=action_to_command)),
            ],
            condition=lambda x: "use_honk" in x[-1].content.lower(),
        ),
        Executor(
            action=EventReportSaver(
                namespace="",
                image_idx=3,
                action_idx=6,
                position=SingleMessageGrabber(
                    topic=f"/odom",
                    message_type=Odometry,
                    timeout_sec=10,
                    postprocess=lambda msg: msg.pose.pose.position,
                ).get_data(),
            )
        ),
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
        vendor = OllamaVendor(
            ip_address="10.244.51.231",
            port="11434",
            model="llava",
            logging_level=logging.INFO,
        )
    elif args.vendor == "openai":
        vendor = OpenAIVendor(model="gpt-4o", stream=False, logging_level=logging.INFO)
    elif args.vendor == "awsbedrock":
        vendor = AWSBedrockVendor(
            model="anthropic.claude-3-opus-20240229-v1:0", logging_level=logging.INFO
        )
    else:
        raise ValueError("Invalid vendor argument")

    scenario_runner = ScenarioRunner(get_scenario(), vendor, logging_level=logging.INFO)
    scenario_runner.run()
    scenario_runner.save_to_html()


if __name__ == "__main__":
    main()
