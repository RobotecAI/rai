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

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import coloredlogs
import tomli
from langchain_aws import ChatBedrock
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.tracers.langchain import LangChainTracer
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langsmith import Client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(level="INFO")  # type: ignore


@dataclass
class VendorConfig:
    simple_model: str
    complex_model: str
    embeddings_model: str


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
class OpenAIConfig(ModelConfig):
    base_url: str


@dataclass
class LangfuseConfig:
    use_langfuse: bool
    host: str


@dataclass
class LangsmithConfig:
    use_langsmith: bool
    host: str


@dataclass
class TracingConfig:
    project: str
    langfuse: LangfuseConfig
    langsmith: LangsmithConfig


@dataclass
class RAIConfig:
    vendor: VendorConfig
    aws: AWSConfig
    openai: OpenAIConfig
    ollama: OllamaConfig
    tracing: TracingConfig


def load_config(config_path: Optional[str] = None) -> RAIConfig:
    if config_path is None:
        with open("config.toml", "rb") as f:
            config_dict = tomli.load(f)
    else:
        with open(config_path, "rb") as f:
            config_dict = tomli.load(f)
    return RAIConfig(
        vendor=VendorConfig(**config_dict["vendor"]),
        aws=AWSConfig(**config_dict["aws"]),
        openai=OpenAIConfig(**config_dict["openai"]),
        ollama=OllamaConfig(**config_dict["ollama"]),
        tracing=TracingConfig(
            project=config_dict["tracing"]["project"],
            langfuse=LangfuseConfig(**config_dict["tracing"]["langfuse"]),
            langsmith=LangsmithConfig(**config_dict["tracing"]["langsmith"]),
        ),
    )


def get_llm_model_config_and_vendor(
    model_type: Literal["simple_model", "complex_model"],
    vendor: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Tuple[str, str]:
    config = load_config(config_path)
    if vendor is None:
        if model_type == "simple_model":
            vendor = config.vendor.simple_model
        else:
            vendor = config.vendor.complex_model

    model_config = getattr(config, vendor)
    return model_config, vendor


def get_llm_model(
    model_type: Literal["simple_model", "complex_model"],
    vendor: Optional[str] = None,
    config_path: Optional[str] = None,
    **kwargs: Any,
) -> ChatOpenAI | ChatBedrock | ChatOllama:
    model_config, vendor = get_llm_model_config_and_vendor(
        model_type, vendor, config_path
    )
    model = getattr(model_config, model_type)
    logger.info(f"Initializing {model_type}: Vendor: {vendor}, Model: {model}")
    if vendor == "openai":
        from langchain_openai import ChatOpenAI

        model_config = cast(OpenAIConfig, model_config)

        return ChatOpenAI(model=model, base_url=model_config.base_url, **kwargs)
    elif vendor == "aws":
        from langchain_aws import ChatBedrock

        model_config = cast(AWSConfig, model_config)

        return ChatBedrock(
            model_id=model,
            region_name=model_config.region_name,
            **kwargs,
        )
    elif vendor == "ollama":
        from langchain_ollama import ChatOllama

        model_config = cast(OllamaConfig, model_config)
        return ChatOllama(model=model, base_url=model_config.base_url, **kwargs)
    else:
        raise ValueError(f"Unknown LLM vendor: {vendor}")


def get_llm_model_direct(
    model_name: str,
    vendor: str,
    config_path: Optional[str] = None,
    **kwargs: Any,
) -> ChatOpenAI | ChatBedrock | ChatOllama:
    config = load_config(config_path)
    model_config = getattr(config, vendor)

    logger.info(f"Initializing Model: {model_name}, Vendor: {vendor}")
    if vendor == "openai":
        from langchain_openai import ChatOpenAI

        model_config = cast(OpenAIConfig, model_config)

        return ChatOpenAI(model=model_name, base_url=model_config.base_url, **kwargs)
    elif vendor == "aws":
        from langchain_aws import ChatBedrock

        model_config = cast(AWSConfig, model_config)

        return ChatBedrock(
            model_id=model_name,
            region_name=model_config.region_name,
            **kwargs,
        )
    elif vendor == "ollama":
        from langchain_ollama import ChatOllama

        model_config = cast(OllamaConfig, model_config)
        return ChatOllama(model=model_name, base_url=model_config.base_url, **kwargs)
    else:
        raise ValueError(f"Unknown LLM vendor: {vendor}")


def get_embeddings_model(
    config_path: Optional[str] = None,
    return_kwargs: bool = False,
) -> Embeddings | Tuple[Embeddings, Dict[str, Any]]:
    config = load_config(config_path)
    vendor = config.vendor.embeddings_model

    model_config = getattr(config, vendor)

    logger.info(f"Using embeddings model: {vendor}-{model_config.embeddings_model}")
    if vendor == "openai":
        from langchain_openai import OpenAIEmbeddings

        model_config = cast(OpenAIConfig, model_config)
        embeddings = OpenAIEmbeddings(model=model_config.embeddings_model)
        if return_kwargs:
            c = (
                str(embeddings.__class__)
                .strip("<>")
                .replace("class '", "")
                .replace("'", "")
            )
            return embeddings, {
                "class": c,
                "model": model_config.embeddings_model,
                "vendor": vendor,
            }
        return embeddings

    elif vendor == "aws":
        from langchain_aws import BedrockEmbeddings

        model_config = cast(AWSConfig, model_config)
        embeddings = BedrockEmbeddings(
            model_id=model_config.embeddings_model, region_name=model_config.region_name
        )
        if return_kwargs:
            c = (
                str(embeddings.__class__)
                .strip("<>")
                .replace("class '", "")
                .replace("'", "")
            )
            return embeddings, {
                "class": c,
                "model_id": model_config.embeddings_model,
                "region_name": model_config.region_name,
                "vendor": vendor,
            }
        return embeddings

    elif vendor == "ollama":
        from langchain_ollama import OllamaEmbeddings

        model_config = cast(OllamaConfig, model_config)
        embeddings = OllamaEmbeddings(
            model=model_config.embeddings_model, base_url=model_config.base_url
        )
        if return_kwargs:
            c = (
                str(embeddings.__class__)
                .strip("<>")
                .replace("class '", "")
                .replace("'", "")
            )
            return embeddings, {
                "class": c,
                "model": model_config.embeddings_model,
                "base_url": model_config.base_url,
                "vendor": vendor,
            }
        return embeddings
    else:
        raise ValueError(f"Unknown embeddings vendor: {vendor}")


def get_tracing_callbacks(
    override_use_langfuse: bool = False, override_use_langsmith: bool = False
) -> List[BaseCallbackHandler]:
    config = load_config()
    callbacks: List[BaseCallbackHandler] = []
    if config.tracing.langfuse.use_langfuse or override_use_langfuse:
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

    if config.tracing.langsmith.use_langsmith or override_use_langsmith:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = config.tracing.project
        api_key = os.getenv("LANGCHAIN_API_KEY", None)
        if api_key is None:
            raise ValueError("LANGCHAIN_API_KEY is not set")
        callback = LangChainTracer(
            project_name=config.tracing.project,
            client=Client(api_key=api_key, api_url=config.tracing.langsmith.host),
        )
        callbacks.append(callback)

    return callbacks
