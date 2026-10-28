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

"""Tests for the LiteLLM vendor in RAI initialization.

langchain-litellm is an optional extra, so these tests install a stand-in
module and check what the factory functions pass to it.
"""

import sys
import types
from pathlib import Path

import pytest
import tomli

from rai.initialization import config_initialization, model_initialization

LITELLM_CONFIG_TEMPLATE = """
[vendor]
simple_model = "litellm"
complex_model = "litellm"
embeddings_model = "litellm"

[litellm]
simple_model = "anthropic/claude-haiku-4-5"
complex_model = "bedrock/anthropic.claude-sonnet-4-6"
embeddings_model = "openai/text-embedding-3-small"
"""


class DummyModel:
    """Records the arguments the factory passes to the LiteLLM wrapper."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def write_config(path: Path, config: str) -> Path:
    path.write_text(config, encoding="utf-8")
    return path


@pytest.fixture
def fake_langchain_litellm(monkeypatch):
    module = types.ModuleType("langchain_litellm")
    module.ChatLiteLLM = type("ChatLiteLLM", (DummyModel,), {})
    module.LiteLLMEmbeddings = type("LiteLLMEmbeddings", (DummyModel,), {})
    monkeypatch.setitem(sys.modules, "langchain_litellm", module)
    return module


class TestLiteLLMVendorConfiguration:
    def test_load_config_reads_litellm_section(self, tmp_path):
        config_path = write_config(tmp_path / "config.toml", LITELLM_CONFIG_TEMPLATE)

        config = model_initialization.load_config(str(config_path))

        assert config.litellm.simple_model == "anthropic/claude-haiku-4-5"
        assert config.litellm.complex_model == "bedrock/anthropic.claude-sonnet-4-6"
        assert config.litellm.api_base == ""

    def test_load_config_reads_optional_api_base(self, tmp_path):
        config = LITELLM_CONFIG_TEMPLATE + 'api_base = "http://localhost:11434"\n'
        config_path = write_config(tmp_path / "config.toml", config)

        assert (
            model_initialization.load_config(str(config_path)).litellm.api_base
            == "http://localhost:11434"
        )

    def test_config_template_includes_a_valid_litellm_section(self):
        template = tomli.loads(config_initialization.CONFIG_CONTENT)

        litellm = model_initialization.LiteLLMConfig(**template["litellm"])
        assert "/" in litellm.simple_model and "/" in litellm.complex_model


class TestLiteLLMChatModels:
    def test_simple_and_complex_models_can_use_different_providers(
        self, fake_langchain_litellm, tmp_path
    ):
        config_path = str(
            write_config(tmp_path / "config.toml", LITELLM_CONFIG_TEMPLATE)
        )

        simple = model_initialization.get_llm_model(
            "simple_model", config_path=config_path
        )
        complex_ = model_initialization.get_llm_model(
            "complex_model", config_path=config_path
        )

        assert isinstance(simple, fake_langchain_litellm.ChatLiteLLM)
        assert simple.kwargs["model"] == "anthropic/claude-haiku-4-5"
        assert complex_.kwargs["model"] == "bedrock/anthropic.claude-sonnet-4-6"

    def test_drop_params_is_on_and_credentials_are_left_to_provider_env(
        self, fake_langchain_litellm, tmp_path
    ):
        config_path = str(
            write_config(tmp_path / "config.toml", LITELLM_CONFIG_TEMPLATE)
        )

        model = model_initialization.get_llm_model(
            "simple_model", config_path=config_path
        )

        assert model.kwargs["model_kwargs"] == {"drop_params": True}
        assert "api_base" not in model.kwargs
        assert "api_key" not in model.kwargs

    def test_api_base_and_caller_kwargs_are_forwarded(
        self, fake_langchain_litellm, tmp_path
    ):
        config = LITELLM_CONFIG_TEMPLATE + 'api_base = "http://localhost:11434"\n'
        config_path = str(write_config(tmp_path / "config.toml", config))

        model = model_initialization.get_llm_model(
            "simple_model",
            config_path=config_path,
            streaming=True,
            model_kwargs={"drop_params": False, "top_k": 5},
        )

        assert model.kwargs["api_base"] == "http://localhost:11434"
        assert model.kwargs["streaming"] is True
        assert model.kwargs["model_kwargs"] == {"drop_params": False, "top_k": 5}

    def test_get_llm_model_direct_uses_the_given_model(
        self, fake_langchain_litellm, tmp_path
    ):
        config_path = str(
            write_config(tmp_path / "config.toml", LITELLM_CONFIG_TEMPLATE)
        )

        model = model_initialization.get_llm_model_direct(
            "vertex_ai/gemini-2.5-flash", vendor="litellm", config_path=config_path
        )

        assert isinstance(model, fake_langchain_litellm.ChatLiteLLM)
        assert model.kwargs["model"] == "vertex_ai/gemini-2.5-flash"
        assert model.kwargs["model_kwargs"] == {"drop_params": True}

    def test_missing_extra_names_the_install_command(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "langchain_litellm", None)
        config_path = str(
            write_config(tmp_path / "config.toml", LITELLM_CONFIG_TEMPLATE)
        )

        with pytest.raises(ImportError, match=r"rai_core\[litellm\]"):
            model_initialization.get_llm_model("simple_model", config_path=config_path)


class TestLiteLLMEmbeddings:
    def test_get_embeddings_model_litellm(self, fake_langchain_litellm, tmp_path):
        config_path = str(
            write_config(tmp_path / "config.toml", LITELLM_CONFIG_TEMPLATE)
        )

        embeddings, kwargs = model_initialization.get_embeddings_model(
            config_path=config_path, return_kwargs=True
        )

        assert isinstance(embeddings, fake_langchain_litellm.LiteLLMEmbeddings)
        assert embeddings.kwargs == {"model": "openai/text-embedding-3-small"}
        assert kwargs["model"] == "openai/text-embedding-3-small"
        assert kwargs["vendor"] == "litellm"
        assert kwargs["class"].endswith("LiteLLMEmbeddings")

    def test_embeddings_forward_api_base(self, fake_langchain_litellm, tmp_path):
        config = LITELLM_CONFIG_TEMPLATE.replace(
            'embeddings_model = "openai/text-embedding-3-small"',
            'embeddings_model = "ollama/nomic-embed-text"\napi_base = "http://localhost:11434"',
        )
        config_path = str(write_config(tmp_path / "config.toml", config))

        embeddings = model_initialization.get_embeddings_model(config_path=config_path)

        assert embeddings.kwargs == {
            "model": "ollama/nomic-embed-text",
            "api_base": "http://localhost:11434",
        }
