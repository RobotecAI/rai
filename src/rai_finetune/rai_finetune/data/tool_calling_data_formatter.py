# Copyright (C) 2025 RAI Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Julia Jia
"""
Simplified training data formatter for tool calling fine-tuning
Focuses on clean, reliable tool calling data without complex multimodal handling
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ToolCallingDataFormatter:
    """Simplified formatter focused on tool calling"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_prompt = config.get(
            "system_prompt",
            "You are a helpful AI assistant that can use tools to help users.",
        )

    def clean_text_content(self, content: Any) -> str:
        """Extract clean text content from various input types"""
        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, list):
            # Handle multimodal content - extract only text
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        # Skip images for now - focus on text
                        continue
            return " ".join(text_parts).strip()
        elif isinstance(content, dict):
            # Convert dict to string representation
            return str(content).strip()
        else:
            return str(content).strip()

    def extract_tool_calls(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from message, handling different formats"""
        tool_calls = []

        # Check for tool_calls in additional_kwargs (GPT-4o format)
        if (
            "additional_kwargs" in message
            and "tool_calls" in message["additional_kwargs"]
        ):
            tool_calls.extend(message["additional_kwargs"]["tool_calls"])

        # Check for direct tool_calls field
        if "tool_calls" in message:
            tool_calls.extend(message["tool_calls"])

        return tool_calls

    def format_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single tool call consistently"""
        function_info = tool_call.get("function", {})

        return {
            "id": tool_call.get("id", f"call_{hash(str(tool_call)) % 10000}"),
            "type": "function",
            "function": {
                "name": function_info.get("name", ""),
                "arguments": function_info.get("arguments", "{}"),
            },
        }

    def should_include_conversation(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if conversation should be included in training data"""
        if len(messages) < 2:
            return False

        # Must have at least one user message and one assistant message
        has_user = any(m.get("role") == "user" for m in messages)
        has_assistant = any(m.get("role") == "assistant" for m in messages)

        if not (has_user and has_assistant):
            return False

        # Check if any assistant message has tool calls
        has_tool_calls = False
        for message in messages:
            if message.get("role") == "assistant":
                tool_calls = self.extract_tool_calls(message)
                if tool_calls:
                    has_tool_calls = True
                    break

        # Only include conversations with tool calls
        return has_tool_calls

    def format_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format a single message for training"""
        role = message.get("role", "")
        content = self.clean_text_content(message.get("content", ""))

        if role == "system":
            return {"role": "system", "content": self.system_prompt}

        elif role == "user":
            if not content:
                return None
            return {"role": "user", "content": content}

        elif role == "assistant":
            tool_calls = self.extract_tool_calls(message)

            # Create assistant message
            formatted_message = {"role": "assistant"}

            # Add content if present
            if content:
                formatted_message["content"] = content
            else:
                formatted_message["content"] = ""

            # Add tool calls if present
            if tool_calls:
                formatted_message["tool_calls"] = [
                    self.format_tool_call(tc) for tc in tool_calls
                ]

            return formatted_message

        elif role == "tool":
            # Convert tool responses to simple text
            if not content:
                return None

            # Skip function definition messages
            if "type" in content and "function" in content:
                return None

            return {"role": "tool", "content": content}

        return None

    def process_conversation(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Process a conversation into training format"""
        if not self.should_include_conversation(messages):
            return None

        formatted_messages = []

        for message in messages:
            formatted = self.format_message(message)
            if formatted:
                formatted_messages.append(formatted)

        # Must have at least 2 messages after formatting
        if len(formatted_messages) < 2:
            return None

        return formatted_messages

    def convert_observations_to_training_data(
        self, input_file: str, output_file: str
    ) -> bool:
        """Convert observations to training data"""
        try:
            logger.info(f"Processing {input_file}...")

            if not os.path.exists(input_file):
                logger.error(f"Input file not found: {input_file}")
                return False

            training_examples = []

            with open(input_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        observation = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                        continue

                    # Process input field (conversation)
                    if "input" in observation:
                        messages = observation["input"]
                        if isinstance(messages, list):
                            formatted = self.process_conversation(messages)
                            if formatted:
                                training_examples.append({"messages": formatted})
                                logger.debug(
                                    f"Line {line_num}: Added input conversation with {len(formatted)} messages"
                                )

                    # Process output field (model response)
                    if "output" in observation:
                        output = observation["output"]
                        if (
                            isinstance(output, dict)
                            and output.get("role") == "assistant"
                        ):
                            # Create a training example from the output
                            # We need to reconstruct the conversation context
                            if "input" in observation and isinstance(
                                observation["input"], list
                            ):
                                # Find the last user message to create context
                                user_messages = [
                                    m
                                    for m in observation["input"]
                                    if m.get("role") == "user"
                                ]
                                if user_messages:
                                    last_user_message = user_messages[-1]

                                    # Create a simple conversation: user -> assistant with tool calls
                                    conversation = [
                                        {
                                            "role": "system",
                                            "content": self.system_prompt,
                                        },
                                        {
                                            "role": "user",
                                            "content": self.clean_text_content(
                                                last_user_message.get("content", "")
                                            ),
                                        },
                                        {
                                            "role": "assistant",
                                            "content": self.clean_text_content(
                                                output.get("content", "")
                                            ),
                                            "tool_calls": self.extract_tool_calls(
                                                output
                                            ),
                                        },
                                    ]

                                    # Only add if it has tool calls
                                    if any(
                                        m.get("tool_calls")
                                        for m in conversation
                                        if m.get("role") == "assistant"
                                    ):
                                        training_examples.append(
                                            {"messages": conversation}
                                        )
                                        logger.debug(
                                            f"Line {line_num}: Added output conversation with tool calls"
                                        )

            logger.info(f"Generated {len(training_examples)} training examples")

            if not training_examples:
                logger.error("No training examples generated")
                return False

            # Save training data
            os.makedirs(
                os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
                exist_ok=True,
            )

            with open(output_file, "w", encoding="utf-8") as f:
                for example in training_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")

            logger.info(f"Saved training data to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to convert observations: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Convert observations to simplified tool calling training data"
    )

    parser.add_argument(
        "--input",
        "-i",
        default="gpt4o_observations.jsonl",
        help="Input observations file (default: gpt4o_observations.jsonl)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="simple_tool_training_data.jsonl",
        help="Output training data file (default: simple_tool_training_data.jsonl)",
    )

    parser.add_argument(
        "--system-prompt",
        "-s",
        default="You are a helpful AI assistant that can use tools to help users.",
        help="System prompt to use for all conversations",
    )

    args = parser.parse_args()

    # Configuration
    config = {"system_prompt": args.system_prompt}

    # Create formatter and run conversion
    formatter = ToolCallingDataFormatter(config)
    success = formatter.convert_observations_to_training_data(args.input, args.output)

    if success:
        logger.info("Conversion completed successfully!")
        return 0
    else:
        logger.error("❌ Conversion failed!")
        return 1


if __name__ == "__main__":
    exit(main())
