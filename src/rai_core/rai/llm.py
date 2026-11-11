# Copyright (C) 2025 Robotec.AI
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

"""Chat and embedding models from `"provider:model"` strings, plus tracing callbacks."""

import functools
import importlib.util
import logging
import os
from typing import TYPE_CHECKING, Any, Literal

from rai.settings import RAISettings, get_settings

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Short names accepted on top of LangChain's provider names.
PROVIDER_ALIASES = {"google": "google_genai", "aws": "bedrock_converse"}

# provider -> (python package, rai_core extra that installs it)
PROVIDER_EXTRAS = {
    "openai": ("langchain_openai", "openai"),
    "azure_openai": ("langchain_openai", "openai"),
    "anthropic": ("langchain_anthropic", "anthropic"),
    "bedrock": ("langchain_aws", "aws"),
    "bedrock_converse": ("langchain_aws", "aws"),
    "google_genai": ("langchain_google_genai", "google"),
    "ollama": ("langchain_ollama", "ollama"),
}


def parse_model(spec: str) -> tuple[str, str]:
    """Split `"provider:model"` on the first colon (`"ollama:llama3.1:70b"` works)."""
    provider, sep, model = spec.partition(":")
    if not sep or not provider or not model:
        raise ValueError(
            f"Model {spec!r} is not a 'provider:model' string, e.g. 'openai:gpt-4o'"
        )
    return PROVIDER_ALIASES.get(provider, provider), model


def _require_provider(provider: str) -> None:
    package, extra = PROVIDER_EXTRAS.get(provider, (None, None))
    if package is not None and importlib.util.find_spec(package) is None:
        raise ImportError(
            f"Provider {provider!r} needs {package}. "
            f"Install it with `pip install rai_core[{extra}]`."
        )


def _resolve(ref: str, settings: RAISettings | None) -> tuple[str, str, dict[str, Any]]:
    models = (settings or get_settings()).models
    if ref in ("simple", "complex", "embeddings"):
        spec = getattr(models, ref)
    else:
        spec = models.aliases.get(ref, ref)
    provider, model = parse_model(spec)
    _require_provider(provider)
    return provider, model, dict(models.provider_kwargs.get(provider, {}))


def get_llm(
    role: Literal["simple", "complex"] | str = "complex",
    *,
    settings: RAISettings | None = None,
    **overrides: Any,
) -> "BaseChatModel":
    """Chat model for a role (`simple`, `complex`, an alias) or a `"provider:model"` string.

    Kwargs come from `models.provider_kwargs[provider]`, then `overrides`.
    """
    from langchain.chat_models import init_chat_model

    provider, model, kwargs = _resolve(role, settings)
    logger.info(f"Initializing chat model {provider}:{model}")
    return init_chat_model(model, model_provider=provider, **(kwargs | overrides))


def get_embeddings(
    ref: str = "embeddings",
    *,
    settings: RAISettings | None = None,
    **overrides: Any,
) -> "Embeddings":
    """Embeddings model for `ref` (`embeddings`, an alias, or `"provider:model"`)."""
    from langchain.embeddings import init_embeddings

    provider, model, kwargs = _resolve(ref, settings)
    logger.info(f"Initializing embeddings {provider}:{model}")
    return init_embeddings(model, provider=provider, **(kwargs | overrides))


@functools.cache
def get_tracing_callbacks() -> list["BaseCallbackHandler"]:
    """Tracing callbacks from `settings.tracing`, built once per process."""
    tracing = get_settings().tracing
    callbacks: list[BaseCallbackHandler] = []
    if tracing.langfuse:
        if importlib.util.find_spec("langfuse") is None:
            raise ImportError(
                "Langfuse tracing needs langfuse. Install it with `pip install rai_core[langfuse]`."
            )
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
            raise ValueError("LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY is not set")
        # configures the singleton client that CallbackHandler uses
        Langfuse(base_url=tracing.langfuse_host)
        callbacks.append(CallbackHandler())
    if tracing.langsmith:
        from langchain_core.tracers.langchain import LangChainTracer
        from langsmith import Client

        callbacks.append(
            LangChainTracer(
                project_name=tracing.project,
                client=Client(api_url=tracing.langsmith_host),
            )
        )
    return callbacks
