from enum import Enum
from typing import TypedDict


class Region(Enum):
    US_WEST_1 = "us-west-1"
    US_WEST_2 = "us-west-2"


class OpenAIKwargs(TypedDict):
    """Dictionary type to specify kwargs for OpenAI models."""

    model: str


class AWSBedrockKwargs(TypedDict):
    """Dictionary type to specify kwargs for AWS Bedrock models."""

    model_id: str
    region_name: Region


# OpenAI Models
OPENAI_MULTIMODAL: OpenAIKwargs = {
    "model": "gpt-4o",
}
OPENAI_LLM: OpenAIKwargs = {
    "model": "gpt-3.5-turbo",
}

# AWS Bedrock Models
BEDROCK_CLAUDE_HAIKU: AWSBedrockKwargs = {
    "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
    "region_name": Region.US_WEST_2,
}

BEDROCK_CLAUDE_SONNET: AWSBedrockKwargs = {
    "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "region_name": Region.US_WEST_1,
}

BEDROCK_CLAUDE_OPUS: AWSBedrockKwargs = {
    "model_id": "anthropic.claude-3-opus-20240229-v1:0",
    "region_name": Region.US_WEST_2,
}
