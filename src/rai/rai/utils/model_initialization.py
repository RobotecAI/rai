# Copyright (C) 2024 Robotec.AI
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

import os
from dataclasses import dataclass
from typing import List, Literal

import tomli
from langchain_core.callbacks.base import BaseCallbackHandler


@dataclass
class VendorConfig:
    name: str


@dataclass
class ModelConfig:
    simple_model: str
    complex_model: str
    embeddings_model: str


@dataclass
class AWSConfig(ModelConfig):
    region_name: str


@dataclass
class OllamaConfig(ModelConfig):
    base_url: str


@dataclass
class LangfuseConfig:
    use_langfuse: bool
    host: str


@dataclass
class LangsmithConfig:
    use_langsmith: bool


@dataclass
class TracingConfig:
    project: str
    langfuse: LangfuseConfig
    langsmith: LangsmithConfig


@dataclass
class RAIConfig:
    vendor: VendorConfig
    aws: AWSConfig
    openai: ModelConfig
    ollama: OllamaConfig
    tracing: TracingConfig


def load_config() -> RAIConfig:
    with open("config.toml", "rb") as f:
        config_dict = tomli.load(f)
    return RAIConfig(
        vendor=VendorConfig(**config_dict["vendor"]),
        aws=AWSConfig(**config_dict["aws"]),
        openai=ModelConfig(**config_dict["openai"]),
        ollama=OllamaConfig(**config_dict["ollama"]),
        tracing=TracingConfig(
            project=config_dict["tracing"]["project"],
            langfuse=LangfuseConfig(**config_dict["tracing"]["langfuse"]),
            langsmith=LangsmithConfig(**config_dict["tracing"]["langsmith"]),
        ),
    )


def get_llm_model(model_type: Literal["simple_model", "complex_model"]):
    config = load_config()
    vendor = config.vendor.name
    model_config = getattr(config, vendor)

    if vendor == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=getattr(model_config, model_type))
    elif vendor == "aws":
        from langchain_aws import ChatBedrock

        return ChatBedrock(
            model_id=getattr(model_config, model_type),
            region_name=model_config.region_name,
        )
    elif vendor == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=getattr(model_config, model_type), base_url=model_config.base_url
        )
    else:
        raise ValueError(f"Unknown LLM vendor: {vendor}")


def get_embeddings_model():
    config = load_config()
    vendor = config.vendor.name
    model_config = getattr(config, vendor)

    if vendor == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model_config.embeddings_model)
    elif vendor == "aws":
        from langchain_aws import BedrockEmbeddings

        return BedrockEmbeddings(
            model_id=model_config.embeddings_model, region_name=model_config.region_name
        )
    elif vendor == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=model_config.embeddings_model, base_url=model_config.base_url
        )
    else:
        raise ValueError(f"Unknown embeddings vendor: {vendor}")


def get_tracing_callbacks() -> List[BaseCallbackHandler]:
    config = load_config()
    callbacks: List[BaseCallbackHandler] = []
    if config.tracing.langfuse.use_langfuse:
        from langfuse.callback import CallbackHandler  # type: ignore

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", None)
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", None)
        if public_key is None or secret_key is None:
            raise ValueError("LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY is not set")

        callback = CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=config.tracing.langfuse.host,
        )
        callbacks.append(callback)

    if config.tracing.langsmith.use_langsmith:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = config.tracing.project
        api_key = os.getenv("LANGCHAIN_API_KEY", None)
        if api_key is None:
            raise ValueError("LANGCHAIN_API_KEY is not set")
    return callbacks
