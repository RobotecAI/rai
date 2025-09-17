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
Base class for training data formatters. This class formats the data samples in ChatML format which unsloth supports natively, see [reference](https://docs.unsloth.ai/basics/datasets-guide#applying-chat-templates-with-unsloth).

Data flow:
Raw Data samples → format_data_sample() → Structured Messages in ChatML format → Final Text
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class DataFormatterConfig:
    """Configuration for data formatters"""

    system_prompt: str = (
        "You are a helpful AI assistant that can use tools to help users."
    )


class BaseDataFormatter(ABC):
    """Base class for training data formatters"""

    def __init__(self, config: Union[Dict[str, Any], DataFormatterConfig]):
        if isinstance(config, dict):
            self.config = DataFormatterConfig(**config)
        else:
            self.config = config

        self.system_prompt = self.config.system_prompt

    def _extract_text_content(self, content: Union[str, list, dict]) -> str:
        """Extract text content from mixed content types.

        Args:
            content: Message content (string, list, or dict)

        Returns:
            Extracted text content as string
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        text_parts.append(item["text"])
                    elif item.get("type") == "image_url":
                        # Replace image with placeholder for text-only training
                        text_parts.append("[IMAGE]")
                    else:
                        # Handle other content types
                        text_parts.append(str(item))
                else:
                    text_parts.append(str(item))
            return " ".join(text_parts)
        elif isinstance(content, dict):
            if content.get("type") == "text" and "text" in content:
                return content["text"]
            else:
                return str(content)
        else:
            return str(content)

    def _normalize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single message to ensure content is string-based.

        Args:
            message: Message dictionary

        Returns:
            Normalized message with string content
        """
        normalized = message.copy()
        if "content" in message:
            normalized["content"] = self._extract_text_content(message["content"])
        return normalized

    @abstractmethod
    def should_include_data_sample(self, data_sample: Dict[str, Any]) -> bool:
        """Check if data sample should be included in training data, for example, a data sample may be excluded if:
        - it has no sufficient conversation data
        - it has no assistant message
        - it has no user message
        - it has no tool calls
        """
        pass

    @abstractmethod
    def format_data_sample(
        self, data_sample: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Format a single data sample for training"""
        pass

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

    def process_data_sample(
        self, data_sample: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process a data sample into training format"""
        if not self.should_include_data_sample(data_sample):
            return None

        formatted = self.format_data_sample(data_sample)
        return formatted

    def convert_data_to_training_data(
        self, input_data_samples_file: str, output_training_data_file: str
    ) -> bool:
        """Convert input data to training data format"""
        logger.info(f"Processing {input_data_samples_file}...")

        training_samples = []

        with open(input_data_samples_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                data_sample = json.loads(line.strip())
                formatted = self.process_data_sample(data_sample)
                if formatted:
                    training_samples.append(formatted)

        logger.info(f"Generated {len(training_samples)} training samples")

        # Save training data
        self._save_training_data(training_samples, output_training_data_file)
        return True

    def _save_training_data(
        self,
        training_samples: List[Dict[str, Any]],
        output_file: str,
    ) -> None:
        """Save training data"""
        # Create output directory
        output_dir = (
            os.path.dirname(output_file) if os.path.dirname(output_file) else "."
        )
        os.makedirs(output_dir, exist_ok=True)

        # Save training data
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(
            f"💾 Saved {len(training_samples)} training samples to {output_file}"
        )
