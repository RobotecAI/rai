# Copyright (C) 2025 Julia Jia
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


"""
Simplified training data formatter for tool calling fine-tuning
Focuses on clean, reliable tool calling data without complex multimodal handling
"""

import argparse
import logging
from typing import Any, Dict, Optional, Union

from .base import BaseDataFormatter, DataFormatterConfig

logger = logging.getLogger(__name__)


class ToolCallingDataFormatter(BaseDataFormatter):
    """Simplified formatter focused on tool calling"""

    def __init__(self, config: Union[Dict[str, Any], DataFormatterConfig]):
        super().__init__(config)

    def should_include_data_sample(self, data_sample: Dict[str, Any]) -> bool:
        """Check if data sample should be included in training data"""
        # Check if data sample has conversation data
        if "input" not in data_sample and "output" not in data_sample:
            return False

        # Check input field (conversation)
        if "input" in data_sample:
            messages = data_sample["input"]
            if isinstance(messages, list) and len(messages) >= 2:
                # Must have at least one user message and one assistant message
                has_user = any(m.get("role") == "user" for m in messages)
                has_assistant = any(m.get("role") == "assistant" for m in messages)

                if has_user and has_assistant:
                    # Check if any assistant message has tool calls
                    for message in messages:
                        if message.get("role") == "assistant":
                            tool_calls = super().extract_tool_calls(message)
                            if tool_calls:
                                return True

        # Check output field (model response)
        if "output" in data_sample:
            output = data_sample["output"]
            if isinstance(output, dict) and output.get("role") == "assistant":
                tool_calls = super().extract_tool_calls(output)
                if tool_calls:
                    return True

        return False

    def format_data_sample(
        self, data_sample: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Format data sample for SFT tool calling training.
        Both input and output are required.
        """

        # SFT requires both input (context) and output (target)
        if "input" not in data_sample or "output" not in data_sample:
            return None

        # Validate input structure
        input_messages = data_sample["input"]
        if not isinstance(input_messages, list) or len(input_messages) < 1:
            return None

        # Validate output structure (must have tool calls)
        output = data_sample["output"]
        if not isinstance(output, dict) or output.get("role") != "assistant":
            return None

        tool_calls = super().extract_tool_calls(output)
        if not tool_calls:  # Must have tool calls for tool calling training
            return None

        # Normalize all messages to text-only content using base class methods
        normalized_input = [self._normalize_message(msg) for msg in input_messages]
        normalized_output = self._normalize_message(output)

        # Combine input + output into complete conversation
        conversation = normalized_input + [normalized_output]

        # Handle system prompt properly
        if conversation and conversation[0].get("role") == "system":
            # Create new system message instead of modifying existing one
            system_message = {"role": "system", "content": self.system_prompt}
            conversation = [system_message] + conversation[1:]  # Replace first message
        else:
            # Add new system message at the beginning
            system_message = {"role": "system", "content": self.system_prompt}
            conversation = [system_message] + conversation

        return {"messages": conversation}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Convert data samples to simplified tool calling training data"
    )

    parser.add_argument(
        "--input",
        "-i",
        default="gpt4o_observations.jsonl",
        help="Input data file (default: gpt4o_observations.jsonl)",
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

    parser.add_argument(
        "--system-prompt-file",
        "-f",
        help="Path to file containing custom system prompt (overrides --system-prompt)",
    )

    args = parser.parse_args()

    # Load custom system prompt from file if specified
    system_prompt = args.system_prompt
    if args.system_prompt_file:
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        logger.info(f"Loaded custom system prompt from {args.system_prompt_file}")

    # Configuration
    config = DataFormatterConfig(
        system_prompt=system_prompt,
    )

    # Create formatter and run conversion
    logger.info(f"🚀 Starting conversion with config: {config}")
    formatter = ToolCallingDataFormatter(config)
    success = formatter.convert_data_to_training_data(args.input, args.output)

    if success:
        logger.info("✅ Conversion completed successfully!")
        return 0
    else:
        logger.error("Conversion failed!")
        return 1


if __name__ == "__main__":
    exit(main())
