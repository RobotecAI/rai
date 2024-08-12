from typing import Literal, TypedDict


class OpenAIKwargs(TypedDict):
    """Dictionary type to specify kwargs for OpenAI models."""

    model: str


class AWSBedrockKwargs(TypedDict):
    """Dictionary type to specify kwargs for AWS Bedrock models."""

    model_id: str
    region_name: Literal["us-west-1", "us-west-2"]


# OpenAI Models
OPENAI_MULTIMODAL: OpenAIKwargs = {
    "model": "gpt-4o",
}
OPENAI_LLM: OpenAIKwargs = {
    "model": "gpt-3.5-turbo",
}

OPENAI_GPT_4o: OpenAIKwargs = {
    "model": "gpt-4o",
}

OPENAI_GPT_3_5_TURBO: OpenAIKwargs = {
    "model": "gpt-3.5-turbo",
}

# AWS Bedrock Models
BEDROCK_CLAUDE_HAIKU: AWSBedrockKwargs = {
    "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
    "region_name": "us-west-2",
}

BEDROCK_CLAUDE_SONNET: AWSBedrockKwargs = {
    "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "region_name": "us-east-1",
}

BEDROCK_CLAUDE_OPUS: AWSBedrockKwargs = {
    "model_id": "anthropic.claude-3-opus-20240229-v1:0",
    "region_name": "us-west-2",
}

BEDROCK_MULTIMODAL: AWSBedrockKwargs = {
    "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "region_name": "us-east-1",
}
