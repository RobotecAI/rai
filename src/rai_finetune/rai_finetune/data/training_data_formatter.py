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
Convert RAI observation JSON files to formatted prompts for LLM finetuning with Unsloth
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingDataConfig:
    """Configuration for training data formatting"""

    input_file: str = "observations.jsonl"
    output_file: str = "training_data.jsonl"
    output_format: str = "unsloth"  # "unsloth" (uses ChatML format), "llama", "chatml"

    # Filtering options
    min_tokens: int = 1  # Reduced from 50 to allow short user messages
    max_tokens: int = 4000
    include_failed_tool_calls: bool = False
    include_image_observations: bool = True

    # Formatting options
    system_prompt_template: str = """
        You are an AI agent deployed on a versatile robotic arm designed for industrial tasks like dispensing,
        remote tool handling, and welding, known for its high precision and adaptability in complex operations.
        """
    clean_tool_responses: bool = True
    remove_metadata: bool = True

    # Output options
    split_conversations: bool = True
    max_conversation_length: int = 20
    include_trace_id: bool = False

    # Training data quality options
    filter_tool_heavy_conversations: bool = True
    min_user_messages: int = 1
    min_assistant_messages: int = 1
    max_tool_to_user_ratio: float = 2.0


class ObservationParser:
    """Parse and validate observation data from JSON files"""

    @staticmethod
    def parse_observation_line(line: str) -> Dict[str, Any]:
        """Parse a single JSON line from the observations file"""
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON line: {e}")
            return {}

    @staticmethod
    def validate_observation(obs: Dict[str, Any]) -> bool:
        """Validate that an observation has required fields"""
        required_fields = ["model", "input", "output"]
        return all(field in obs for field in required_fields)

    @staticmethod
    def extract_conversation(obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract conversation messages from observation input"""
        if "input" not in obs:
            return []

        input_data = obs["input"]
        if isinstance(input_data, list):
            return input_data
        elif isinstance(input_data, dict):
            return [input_data]
        else:
            return []


class MessageFormatter:
    """Format messages for different training data formats"""

    @staticmethod
    def format_system_message(
        content: str, config: TrainingDataConfig
    ) -> Dict[str, str]:
        """Format a system message"""
        return {"role": "system", "content": config.system_prompt_template}

    @staticmethod
    def format_user_message(
        content: Union[str, List, Dict], config: TrainingDataConfig
    ) -> Dict[str, str]:
        """Format a user message"""
        if isinstance(content, list):
            # Handle multimodal content
            text_content = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_content.append(item.get("text", ""))
                    elif (
                        item.get("type") == "image_url"
                        and config.include_image_observations
                    ):
                        text_content.append(
                            f"[Image: {item.get('image_url', {}).get('url', '')}]"
                        )

            content = " ".join(text_content)
        elif isinstance(content, dict):
            # Handle dictionary content (e.g., function calls)
            content = str(content)

        return {"role": "user", "content": str(content).strip()}

    @staticmethod
    def format_assistant_message(
        content: Union[str, List, Dict],
        tool_calls: List[Dict[str, Any]],
        config: TrainingDataConfig,
    ) -> Dict[str, Any]:
        """Format an assistant message with tool calls"""
        # Handle multimodal content
        if isinstance(content, list):
            text_content = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_content.append(item.get("text", ""))
                    elif (
                        item.get("type") == "image_url"
                        and config.include_image_observations
                    ):
                        text_content.append(
                            f"[Image: {item.get('image_url', {}).get('url', '')}]"
                        )

            content = " ".join(text_content)
        elif isinstance(content, dict):
            # Handle dictionary content (e.g., function calls)
            content = str(content)

        message = {"role": "assistant", "content": content.strip() if content else ""}

        if tool_calls:
            # Format tool calls for Unsloth
            if config.output_format == "unsloth":
                message["tool_calls"] = [
                    {
                        "id": tool_call.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tool_call.get("function", {}).get("name", ""),
                            "arguments": tool_call.get("function", {}).get(
                                "arguments", "{}"
                            ),
                        },
                    }
                    for i, tool_call in enumerate(tool_calls)
                ]
            else:
                # For other formats, include tool calls in content
                tool_call_text = []
                for tool_call in tool_calls:
                    func_name = tool_call.get("function", {}).get("name", "")
                    func_args = tool_call.get("function", {}).get("arguments", "{}")
                    tool_call_text.append(f"Tool call: {func_name}({func_args})")

                if tool_call_text:
                    message["content"] += "\n" + "\n".join(tool_call_text)

        return message

    @staticmethod
    def format_tool_message(
        content: Union[str, List, Dict], config: TrainingDataConfig
    ) -> Dict[str, str]:
        """Format a tool response message"""
        if isinstance(content, list):
            # Handle multimodal tool responses
            text_content = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_content.append(item.get("text", ""))
                    elif (
                        item.get("type") == "image_url"
                        and config.include_image_observations
                    ):
                        text_content.append(
                            f"[Image response: {item.get('image_url', {}).get('url', '')}]"
                        )

            content = " ".join(text_content)
        elif isinstance(content, dict):
            # Handle dictionary content (e.g., function calls)
            content = str(content)

        return {"role": "tool", "content": str(content).strip()}


class ConversationProcessor:
    """Process and clean conversations for training"""

    def __init__(self, config: TrainingDataConfig, formatter: "MessageFormatter"):
        self.config = config
        self.formatter = formatter

    def clean_tool_response(self, content: Union[str, List, Dict]) -> str:
        """Clean tool response content"""
        # Handle multimodal content
        if isinstance(content, list):
            text_content = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_content.append(item.get("text", ""))
                    elif (
                        item.get("type") == "image_url"
                        and self.config.include_image_observations
                    ):
                        text_content.append(
                            f"[Image response: {item.get('image_url', {}).get('url', '')}]"
                        )

            content = " ".join(text_content)
        elif isinstance(content, dict):
            # Handle dictionary content (e.g., function calls)
            content = str(content)

        if not self.config.clean_tool_responses:
            return content

        # Remove function definitions that often appear in tool responses
        content = re.sub(
            r'{"type": "function", "function": \{.*?\}}', "", content, flags=re.DOTALL
        )

        # Clean up extra whitespace
        content = re.sub(r"\s+", " ", content).strip()

        return content

    def should_include_message(self, message: Dict[str, Any]) -> bool:
        """Determine if a message should be included in training data"""
        role = message.get("role", "")
        content = message.get("content", "")

        # Handle multimodal content for length checking
        if isinstance(content, list):
            text_content = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_content.append(item.get("text", ""))
                    elif (
                        item.get("type") == "image_url"
                        and self.config.include_image_observations
                    ):
                        text_content.append(
                            f"[Image: {item.get('image_url', {}).get('url', '')}]"
                        )

            content = " ".join(text_content)
        elif isinstance(content, dict):
            # Handle dictionary content (e.g., function calls)
            content = str(content)
        elif not isinstance(content, str):
            # Convert any other type to string
            content = str(content)

        # Skip completely empty messages
        if not content or content.strip() == "":
            # But allow assistant messages with tool calls even if content is empty
            if role == "assistant" and self.has_tool_calls(message):
                return True
            return False

        # Skip failed tool calls if configured
        if role == "tool" and not self.config.include_failed_tool_calls:
            if "Failed to run tool" in content or "Error:" in content:
                return False

        # For content length checking, be more lenient with certain message types
        content_length = len(content.split())

        # Allow system messages regardless of length
        if role == "system":
            return True

        # Allow user messages with shorter content (they're often just instructions)
        if role == "user" and content_length >= 1:
            return True

        # Allow assistant messages with tool calls regardless of content length
        if role == "assistant" and self.has_tool_calls(message):
            return True

        # For other messages, apply the token length limits
        if (
            content_length < self.config.min_tokens
            or content_length > self.config.max_tokens
        ):
            return False

        return True

    def filter_conversation_for_training(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter conversation to create better training examples"""
        if not self.config.filter_tool_heavy_conversations:
            return messages

        if len(messages) < 2:  # Need at least user + response
            logger.debug(f"Conversation too short: {len(messages)} messages")
            return []

        # Count different message types
        user_count = sum(1 for m in messages if m.get("role") == "user")
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
        system_count = sum(1 for m in messages if m.get("role") == "system")

        logger.debug(
            f"Message counts - System: {system_count}, User: {user_count}, Assistant: {assistant_count}"
        )

        # Skip if no user messages
        if user_count == 0:
            logger.debug("No user messages found")
            return []

        # Skip if no assistant responses
        if assistant_count == 0:
            logger.debug("No assistant messages found")
            return []

        # Ensure we have a proper conversation flow
        # Look for patterns like: system->user, user->assistant, assistant->user
        has_good_flow = False
        for i in range(len(messages) - 1):
            current_role = messages[i].get("role")
            next_role = messages[i + 1].get("role")

            # Good patterns: system->user, user->assistant, assistant->user
            if (
                (current_role == "system" and next_role == "user")
                or (current_role == "user" and next_role == "assistant")
                or (current_role == "assistant" and next_role == "user")
            ):
                has_good_flow = True
                break

        if not has_good_flow and len(messages) > 2:
            logger.debug("Poor conversation flow detected")
            return []

        logger.debug(f"Conversation passed filtering with {len(messages)} messages")
        return messages

    def create_training_examples(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create better training examples by restructuring conversations"""
        if len(messages) < 2:
            return []

        # Group messages into logical conversation turns
        conversation_turns = []
        current_turn = []

        for message in messages:
            role = message.get("role")

            if role == "system":
                # System message starts a new conversation
                if current_turn:
                    conversation_turns.append(current_turn)
                current_turn = [message]
            elif role == "user":
                # User message starts a new turn
                if current_turn and current_turn[-1].get("role") == "assistant":
                    # Previous turn is complete
                    conversation_turns.append(current_turn)
                current_turn = [message]
            elif role == "assistant":
                # Assistant message completes the turn
                current_turn.append(message)
            elif role == "tool":
                # Tool responses can be converted to assistant messages
                if current_turn and current_turn[-1].get("role") == "assistant":
                    # Add tool response as part of assistant's response
                    tool_content = f"Tool response: {message.get('content', '')}"
                    current_turn[-1]["content"] += f"\n\n{tool_content}"

        # Add the last turn if it exists
        if current_turn:
            conversation_turns.append(current_turn)

        # Create training examples from conversation turns
        training_examples = []
        for turn in conversation_turns:
            if len(turn) >= 2:  # Need at least user + assistant
                # Ensure proper role sequence
                if turn[0].get("role") == "user" and turn[1].get("role") == "assistant":
                    training_examples.append({"messages": turn})
                elif turn[0].get("role") == "system" and len(turn) >= 3:
                    # System + user + assistant
                    if (
                        turn[1].get("role") == "user"
                        and turn[2].get("role") == "assistant"
                    ):
                        training_examples.append({"messages": turn})

        return training_examples

    def has_tool_calls(self, message: Dict[str, Any]) -> bool:
        """Check if a message has tool calls"""
        # Check for tool_calls in additional_kwargs
        if (
            "additional_kwargs" in message
            and "tool_calls" in message["additional_kwargs"]
        ):
            return len(message["additional_kwargs"]["tool_calls"]) > 0

        # Check for direct tool_calls field
        if "tool_calls" in message:
            return len(message["tool_calls"]) > 0

        return False

    def extract_tool_calls(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from a message"""
        tool_calls = []

        # Check for tool_calls in additional_kwargs
        if (
            "additional_kwargs" in message
            and "tool_calls" in message["additional_kwargs"]
        ):
            tool_calls.extend(message["additional_kwargs"]["tool_calls"])

        # Check for direct tool_calls field
        if "tool_calls" in message:
            tool_calls.extend(message["tool_calls"])

        return tool_calls

    def process_conversation(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process and clean a conversation"""
        # First filter the conversation for better training examples
        filtered_messages = self.filter_conversation_for_training(messages)
        if not filtered_messages:
            return []

        processed_messages = []

        for message in filtered_messages:
            if not self.should_include_message(message):
                continue

            # Format the message based on its role
            role = message.get("role", "")
            content = message.get("content", "")
            tool_calls = self.extract_tool_calls(message)

            try:
                if role == "system":
                    formatted_message = self.formatter.format_system_message(
                        content, self.config
                    )
                elif role == "user":
                    formatted_message = self.formatter.format_user_message(
                        content, self.config
                    )
                elif role == "assistant":
                    formatted_message = self.formatter.format_assistant_message(
                        content, tool_calls, self.config
                    )
                elif role == "tool":
                    formatted_message = self.formatter.format_tool_message(
                        content, self.config
                    )
                else:
                    # Skip unknown role types
                    continue

                # Clean tool responses if needed
                if role == "tool" and self.config.clean_tool_responses:
                    formatted_message["content"] = self.clean_tool_response(
                        formatted_message["content"]
                    )

                # Remove metadata if configured
                if self.config.remove_metadata:
                    formatted_message.pop("additional_kwargs", None)
                    formatted_message.pop("id", None)

                processed_messages.append(formatted_message)

            except Exception as e:
                logger.warning(f"Failed to format message with role '{role}': {e}")
                continue

        return processed_messages

    def split_long_conversations(
        self, messages: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Split long conversations into smaller chunks"""
        if not self.config.split_conversations:
            return [messages]

        conversations = []
        current_conversation = []

        for message in messages:
            current_conversation.append(message)

            if len(current_conversation) >= self.config.max_conversation_length:
                conversations.append(current_conversation)
                current_conversation = []

        if current_conversation:
            conversations.append(current_conversation)

        return conversations


class TrainingDataFormatter:
    """Main class for formatting training data"""

    def __init__(self, config: TrainingDataConfig):
        self.config = config
        self.parser = ObservationParser()
        self.formatter = MessageFormatter()
        self.processor = ConversationProcessor(config, self.formatter)

    def format_for_unsloth(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format messages for Unsloth fine-tuning using ChatML format with preserved tool calls"""
        # Clean and restructure messages for better training
        cleaned_messages = []

        for i, message in enumerate(messages):
            role = message.get("role", "")
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])

            # Skip empty messages
            if not content and not tool_calls:
                continue

            # Handle system messages
            if role == "system":
                cleaned_messages.append({"role": "system", "content": content})
                continue

            # Handle user messages
            if role == "user":
                cleaned_messages.append({"role": "user", "content": content})
                continue

            # Handle assistant messages with tool calls
            if role == "assistant" and tool_calls:
                message_dict = {"role": "assistant"}

                # Add content if present
                if content:
                    message_dict["content"] = content
                else:
                    message_dict["content"] = ""

                # Preserve tool calls in structured format
                formatted_tool_calls = []
                for tool_call in tool_calls:
                    formatted_tool_call = {
                        "id": tool_call.get("id", f"call_{len(formatted_tool_calls)}"),
                        "type": "function",
                        "function": {
                            "name": tool_call.get("function", {}).get("name", ""),
                            "arguments": tool_call.get("function", {}).get(
                                "arguments", "{}"
                            ),
                        },
                    }
                    formatted_tool_calls.append(formatted_tool_call)

                if formatted_tool_calls:
                    message_dict["tool_calls"] = formatted_tool_calls

                cleaned_messages.append(message_dict)
                continue

            # Handle regular assistant messages
            if role == "assistant":
                cleaned_messages.append({"role": "assistant", "content": content})
                continue

            # Handle tool messages - preserve them for proper tool calling flow
            if role == "tool":
                # Skip function definition messages (they're not useful for training)
                if "type" in str(content) and "function" in str(content):
                    continue

                # Preserve tool messages as tool messages, not assistant messages
                tool_message = {"role": "tool", "content": content}

                # Add tool_call_id if available (important for proper tool response linking)
                if "tool_call_id" in message:
                    tool_message["tool_call_id"] = message["tool_call_id"]
                elif "name" in message:
                    tool_message["name"] = message["name"]

                cleaned_messages.append(tool_message)
                continue

        # Ensure we have a proper conversation structure
        if len(cleaned_messages) < 2:
            return None  # Skip conversations that are too short

        # Return in ChatML format for Unsloth
        return {"messages": cleaned_messages}

    def format_for_llama(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages for LLaMA fine-tuning"""
        formatted = ""

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                formatted += f"<|system|>\n{content}\n<|end|>\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}\n<|end|>\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}\n<|end|>\n"
            elif role == "tool":
                formatted += f"<|tool|>\n{content}\n<|end|>\n"

        return formatted

    def format_for_chatml(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages for ChatML fine-tuning"""
        formatted = ""

        for message in messages:
            role = message["role"]
            content = message["content"]

            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        return formatted

    def format_conversation(
        self, messages: List[Dict[str, Any]]
    ) -> Union[Dict[str, Any], str]:
        """Format a conversation based on the configured output format"""
        if self.config.output_format == "unsloth":
            return self.format_for_unsloth(messages)
        elif self.config.output_format == "llama":
            return self.format_for_llama(messages)
        elif self.config.output_format == "chatml":
            return self.format_for_chatml(messages)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")


class TrainingDataPipeline:
    """Main pipeline for converting observations to training data"""

    def __init__(self, config: TrainingDataConfig):
        self.config = config
        self.formatter = TrainingDataFormatter(config)

    def load_observations(self, input_file: str) -> List[Dict[str, Any]]:
        """Load observations from JSONL file"""
        observations = []

        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return []

        with open(input_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    obs = self.formatter.parser.parse_observation_line(line)
                    if obs and self.formatter.parser.validate_observation(obs):
                        observations.append(obs)
                    else:
                        logger.warning(f"Invalid observation at line {line_num}")

        logger.info(f"Loaded {len(observations)} valid observations")
        return observations

    def process_observation(self, obs: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Process a single observation into conversations"""
        try:
            # Extract conversation messages
            messages = self.formatter.parser.extract_conversation(obs)
            if not messages:
                return []

            # Process and clean messages
            processed_messages = self.formatter.processor.process_conversation(messages)

            if not processed_messages:
                logger.debug("No messages passed processing/filtering")
                return []

            logger.debug(f"Processed messages: {len(processed_messages)}")

            # Split into conversations if needed
            conversations = self.formatter.processor.split_long_conversations(
                processed_messages
            )

            logger.debug(f"Created {len(conversations)} conversations")
            return conversations
        except Exception as e:
            logger.warning(f"Failed to process observation: {e}")
            return []

    def save_training_data(self, training_data: List[Any], output_file: str):
        """Save formatted training data to file"""
        os.makedirs(
            os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
            exist_ok=True,
        )

        with open(output_file, "w", encoding="utf-8") as f:
            for item in training_data:
                if self.config.output_format == "unsloth":
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                else:
                    f.write(item + "\n")

        logger.info(f"Saved {len(training_data)} training examples to {output_file}")

    def run(self) -> bool:
        """Run the complete training data conversion pipeline"""
        try:
            logger.info("Starting training data conversion pipeline...")

            # Load observations
            observations = self.load_observations(self.config.input_file)
            if not observations:
                logger.error("No observations to process")
                return False

            # Process observations into training data
            all_conversations = []
            for i, obs in enumerate(observations):
                conversations = self.process_observation(obs)
                all_conversations.extend(conversations)

            logger.info(f"Total conversations generated: {len(all_conversations)}")

            if not all_conversations:
                logger.error("No valid conversations generated")
                return False

            # Create better training examples
            training_data = []
            for conversation in all_conversations:
                try:
                    # Create training examples from the conversation
                    examples = self.formatter.processor.create_training_examples(
                        conversation
                    )
                    for example in examples:
                        formatted = self.formatter.format_conversation(
                            example["messages"]
                        )
                        if formatted:  # Skip None results
                            training_data.append(formatted)
                except Exception as e:
                    logger.warning(f"Failed to format conversation: {e}")
                    continue

            if not training_data:
                logger.error("No training data generated after formatting")
                return False

            # Save training data
            self.save_training_data(training_data, self.config.output_file)

            logger.info(
                f"Successfully converted {len(observations)} observations to {len(training_data)} training examples"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Convert RAI observations to training data"
    )

    parser.add_argument(
        "--input",
        "-i",
        default="observations.jsonl",
        help="Input observations file (default: observations.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="training_data.jsonl",
        help="Output training data file (default: training_data.jsonl)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["unsloth", "llama", "chatml"],
        default="unsloth",
        help="Output format: unsloth (ChatML format), llama, or chatml (default: unsloth)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=50,
        help="Minimum tokens per message (default: 50)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Maximum tokens per message (default: 4000)",
    )
    parser.add_argument(
        "--include-failed-tools",
        action="store_true",
        help="Include failed tool calls in training data",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Exclude image observations from training data",
    )
    parser.add_argument(
        "--max-conversation-length",
        type=int,
        default=20,
        help="Maximum messages per conversation (default: 20)",
    )
    parser.add_argument(
        "--system-prompt", default="", help="Custom system prompt template"
    )
    parser.add_argument(
        "--no-filter-tool-heavy",
        action="store_true",
        help="Don't filter out tool-heavy conversations",
    )
    parser.add_argument(
        "--min-user-messages",
        type=int,
        default=1,
        help="Minimum user messages per conversation (default: 1)",
    )
    parser.add_argument(
        "--min-assistant-messages",
        type=int,
        default=1,
        help="Minimum assistant messages per conversation (default: 1)",
    )
    parser.add_argument(
        "--max-tool-ratio",
        type=float,
        default=2.0,
        help="Maximum tool messages per user message (default: 2.0)",
    )

    args = parser.parse_args()

    # Create configuration
    config = TrainingDataConfig(
        input_file=args.input,
        output_file=args.output,
        output_format=args.format,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        include_failed_tool_calls=args.include_failed_tools,
        include_image_observations=not args.no_images,
        max_conversation_length=args.max_conversation_length,
        filter_tool_heavy_conversations=not args.no_filter_tool_heavy,
        min_user_messages=args.min_user_messages,
        min_assistant_messages=args.min_assistant_messages,
        max_tool_to_user_ratio=args.max_tool_ratio,
    )

    if args.system_prompt:
        config.system_prompt_template = args.system_prompt

    # Run pipeline
    pipeline = TrainingDataPipeline(config)
    success = pipeline.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
