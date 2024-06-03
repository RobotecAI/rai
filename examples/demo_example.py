import argparse
import logging

from rai.actions.actions import SendEmailAction
from rai.actions.executor import Executor
from rai.message import AssistantMessage, ConstantMessage, Message
from rai.requirements import MessageLengthRequirement, RequirementSeverity
from rai.scenario_engine.scenario_engine import ConditionalScenario, ScenarioRunner
from rai.vendors.vendors import AWSBedrockVendor, OllamaVendor, OpenAIVendor

logging.basicConfig(level=logging.INFO)


AUTONOMOUS_TRACTOR_SYSTEM_PROMPT = [
    ConstantMessage(
        role="system",
        content="You are a visual language model designed for ensuring operational safety in autonomous robotics. "
        "Your task is to monitor and interpret visual data in real time to detect potential hazards and assist in decision-making."
        "You are specifically assigned to work with an autonomous tractor. "
        "You will receive images captured by the tractor's camera system. "
        "Focus on analyzing imagery from both near and far fields to identify any obstacles or hazards. "
        "Your interpretation should help determine the most effective strategies to maintain safety and efficiency during the tractor's operation.",
    ),
]

SAFETY_OR_ANOMALY_PROMPT = [
    ConstantMessage(
        role="user",
        content="Does the image indicate a safety concern or anomaly? Reply with one word only: yes or no.",
        images=[],
    ),
    AssistantMessage(
        requirements=[
            MessageLengthRequirement(
                severity=RequirementSeverity.OPTIONAL, max_length=5
            )
        ]
    ),
]

ASK_FOR_DESCRIPTION = [
    ConstantMessage(
        role="user",
        content="Provide a concise description of the identified safety concern or anomaly. The description should be at most 5 words long.",
    ),
    AssistantMessage(),
]

PRESENT_OPTIONS = ConstantMessage(
    role="user",
    content="Please choose one of the following options by replying with a single word representing your choice. \n"
    + "\n".join(
        [
            "change_path",
            "continue",
            "stop",
            "use_honk",
            "visual_signal",
            "stop",
        ]
    ),
)


PRESENT_OPTIONS_AND_ASK_WHAT_TO_DO = [PRESENT_OPTIONS, AssistantMessage()]


scenario = [
    *AUTONOMOUS_TRACTOR_SYSTEM_PROMPT,
    ConstantMessage(
        role="user",
        content="Analyze the provided image from the autonomous tractor's front camera. "
        "Identify and report any unusual, hazardous, or suspicious elements that could impact the tractor's safety or navigation path. "
        "Detail your analysis process and reasoning for any operational recommendations.",
        images=[
            Message.preprocess_image("examples/imgs/barrel_close.png"),
        ],
    ),
    AssistantMessage(),
    *SAFETY_OR_ANOMALY_PROMPT,
    ConditionalScenario(
        if_true=[
            *ASK_FOR_DESCRIPTION,
            Executor(SendEmailAction("email@email.com")),
        ],
        if_false=[
            ConstantMessage(
                role="user",
                content="Based on the analysis, can the tractor safely continue its operation?",
            )
        ],
        condition=lambda x: "yes" in x[-1].content.lower(),
    ),
    *PRESENT_OPTIONS_AND_ASK_WHAT_TO_DO,
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

    scenario_runner = ScenarioRunner(scenario, vendor, logging_level=logging.INFO)
    scenario_runner.run()
    scenario_runner.save_to_html()


if __name__ == "__main__":
    main()
