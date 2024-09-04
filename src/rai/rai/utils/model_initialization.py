from typing import Literal, TypedDict

import tomli


class LLMConfig(TypedDict):
    vendor: str
    simple_model: str
    complex_model: str
    region_name: str
    base_url: str


class EmbeddingsConfig(TypedDict):
    vendor: str
    model: str
    region_name: str
    base_url: str


def get_llm_model(model_type: Literal["simple_model", "complex_model"]):
    rai_config = tomli.load(open("config.toml", "rb"))
    llm_config = LLMConfig(**rai_config["llm"])
    if llm_config["vendor"] == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=llm_config[model_type])
    elif llm_config["vendor"] == "aws":
        from langchain_aws import ChatBedrock

        return ChatBedrock(model_id=llm_config[model_type], region_name=llm_config["region_name"])  # type: ignore
    elif llm_config["vendor"] == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=llm_config[model_type], base_url=llm_config["base_url"])
    else:
        raise ValueError(f"Unknown LLM vendor: {llm_config['vendor']}")


def get_embeddings_model():
    rai_config = tomli.load(open("config.toml", "rb"))
    embeddings_config = EmbeddingsConfig(**rai_config["embeddings"])

    if embeddings_config["vendor"] == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings()
    elif embeddings_config["vendor"] == "aws":
        from langchain_aws import BedrockEmbeddings

        return BedrockEmbeddings(model_id=embeddings_config["model"], region_name=embeddings_config["region_name"])  # type: ignore
    elif embeddings_config["vendor"] == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=embeddings_config["model"], base_url=embeddings_config["base_url"]
        )
    else:
        raise ValueError(f"Unknown embeddings vendor: {embeddings_config['vendor']}")
