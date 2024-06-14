import pytest


@pytest.fixture
def chat_openai_multimodal():
    from langchain_openai.chat_models import ChatOpenAI

    return ChatOpenAI(model="gpt-4o")


@pytest.fixture
def chat_openai_text():
    from langchain_openai.chat_models import ChatOpenAI

    return ChatOpenAI(model="gpt-3.5")


@pytest.fixture
def chat_bedrock_multimodal():
    from langchain_aws.chat_models import ChatBedrock

    return ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-west-2"
    )
