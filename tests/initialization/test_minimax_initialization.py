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

"""Tests for MiniMax vendor integration with RAI framework.

MiniMax exposes an OpenAI-compatible chat API at https://api.minimax.io/v1.
These tests verify that the RAI initialization module correctly routes to
ChatOpenAI with the MiniMax base URL and API key, and that temperature
clamping (>0.0) is applied before construction.
"""

from pathlib import Path

import pytest
from rai.initialization import model_initialization

# ---------------------------------------------------------------------------
# Shared config template
# ---------------------------------------------------------------------------

MINIMAX_CONFIG_TEMPLATE = """
[vendor]
simple_model = "minimax"
complex_model = "minimax"
embeddings_model = "openai"

[aws]
simple_model = "aws.simple"
complex_model = "aws.complex"
embeddings_model = "aws.embed"
region_name = "us-west-2"

[openai]
simple_model = "gpt-4o-mini"
complex_model = "gpt-4o"
embeddings_model = "text-embedding-ada-002"
base_url = "https://api.openai.com/v1/"

[ollama]
simple_model = "llama-simple"
complex_model = "llama-complex"
embeddings_model = "llama-embed"
base_url = "http://localhost:11434"

[google]
simple_model = "gemini-3-flash"
complex_model = "gemini-3-pro"
embeddings_model = "text-embedding-004"

[minimax]
simple_model = "MiniMax-M2.7-highspeed"
complex_model = "MiniMax-M2.7"
base_url = "https://api.minimax.io/v1"

[tracing]
project = "rai"

[tracing.langfuse]
use_langfuse = false
host = "http://localhost:3000"

[tracing.langsmith]
use_langsmith = false
host = "https://api.smith.langchain.com"
"""


class DummyModel:
    """Dummy model for testing factory routing without real API calls."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def write_config(path: Path, config: str) -> Path:
    path.write_text(config, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestMiniMaxConfig:
    """Tests for MiniMax config loading."""

    def test_load_config_includes_minimax_section(self, tmp_path):
        config_path = write_config(tmp_path / "config.toml", MINIMAX_CONFIG_TEMPLATE)
        config = model_initialization.load_config(str(config_path))

        assert config.minimax.simple_model == "MiniMax-M2.7-highspeed"
        assert config.minimax.complex_model == "MiniMax-M2.7"
        assert config.minimax.base_url == "https://api.minimax.io/v1"

    def test_load_config_defaults_when_minimax_section_absent(self, tmp_path):
        """Configs without [minimax] section should use built-in defaults."""
        config_without_minimax = "\n".join(
            line
            for line in MINIMAX_CONFIG_TEMPLATE.splitlines()
            if not line.startswith("[minimax]")
            and not line.startswith('simple_model = "MiniMax')
            and not line.startswith('complex_model = "MiniMax')
            and not line.startswith('base_url = "https://api.minimax')
        )
        config_path = write_config(tmp_path / "config.toml", config_without_minimax)
        config = model_initialization.load_config(str(config_path))

        assert config.minimax.simple_model == "MiniMax-M2.7-highspeed"
        assert config.minimax.complex_model == "MiniMax-M2.7"
        assert config.minimax.base_url == "https://api.minimax.io/v1"

    def test_get_llm_model_config_and_vendor_minimax_simple(self, tmp_path):
        config_path = write_config(tmp_path / "config.toml", MINIMAX_CONFIG_TEMPLATE)

        model_config, vendor = model_initialization.get_llm_model_config_and_vendor(
            "simple_model", config_path=str(config_path)
        )

        assert vendor == "minimax"
        assert model_config.simple_model == "MiniMax-M2.7-highspeed"

    def test_get_llm_model_config_and_vendor_minimax_complex(self, tmp_path):
        config_path = write_config(tmp_path / "config.toml", MINIMAX_CONFIG_TEMPLATE)

        model_config, vendor = model_initialization.get_llm_model_config_and_vendor(
            "complex_model", config_path=str(config_path)
        )

        assert vendor == "minimax"
        assert model_config.complex_model == "MiniMax-M2.7"

    def test_get_llm_model_config_and_vendor_override_to_minimax(self, tmp_path):
        """Explicitly passing vendor='minimax' works regardless of vendor defaults."""
        config_openai_default = MINIMAX_CONFIG_TEMPLATE.replace(
            'simple_model = "minimax"', 'simple_model = "openai"', 1
        )
        config_path = write_config(tmp_path / "config.toml", config_openai_default)

        model_config, vendor = model_initialization.get_llm_model_config_and_vendor(
            "simple_model", vendor="minimax", config_path=str(config_path)
        )

        assert vendor == "minimax"
        assert model_config.simple_model == "MiniMax-M2.7-highspeed"


# ---------------------------------------------------------------------------
# get_llm_model factory
# ---------------------------------------------------------------------------


class TestMiniMaxLLMModelFactory:
    """Tests for get_llm_model factory with MiniMax vendor."""

    def test_get_llm_model_returns_chat_openai_instance(self, monkeypatch, tmp_path):
        config_path = write_config(tmp_path / "config.toml", MINIMAX_CONFIG_TEMPLATE)
        monkeypatch.setattr("langchain_openai.ChatOpenAI", DummyModel)

        model = model_initialization.get_llm_model(
            "simple_model", vendor="minimax", config_path=str(config_path)
        )

        assert isinstance(model, DummyModel)
        assert model.kwargs["model"] == "MiniMax-M2.7-highspeed"
        assert model.kwargs["base_url"] == "https://api.minimax.io/v1"

    def test_get_llm_model_complex_uses_m2_7(self, monkeypatch, tmp_path):
        config_path = write_config(tmp_path / "config.toml", MINIMAX_CONFIG_TEMPLATE)
        monkeypatch.setattr("langchain_openai.ChatOpenAI", DummyModel)

        model = model_initialization.get_llm_model(
            "complex_model", vendor="minimax", config_path=str(config_path)
        )

        assert isinstance(model, DummyModel)
        assert model.kwargs["model"] == "MiniMax-M2.7"

    def test_get_llm_model_reads_minimax_api_key_env_var(self, monkeypatch, tmp_path):
        config_path = write_config(tmp_path / "config.toml", MINIMAX_CONFIG_TEMPLATE)
        monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
        monkeypatch.setattr("langchain_openai.ChatOpenAI", DummyModel)

        model = model_initialization.get_llm_model(
            "simple_model", vendor="minimax", config_path=str(config_path)
        )

        assert model.kwargs["api_key"] == "test-minimax-key"

    def test_get_llm_model_passes_extra_kwargs(self, monkeypatch, tmp_path):
        config_path = write_config(tmp_path / "config.toml", MINIMAX_CONFIG_TEMPLATE)
        monkeypatch.setattr("langchain_openai.ChatOpenAI", DummyModel)

        model = model_initialization.get_llm_model(
            "simple_model",
            vendor="minimax",
            config_path=str(config_path),
            temperature=0.7,
            max_tokens=2048,
        )

        assert model.kwargs["temperature"] == 0.7
        assert model.kwargs["max_tokens"] == 2048

    def test_get_llm_model_clamps_zero_temperature(self, monkeypatch, tmp_path):
        """MiniMax requires temperature > 0.0; zero should be clamped to 0.01."""
        config_path = write_config(tmp_path / "config.toml", MINIMAX_CONFIG_TEMPLATE)
        monkeypatch.setattr("langchain_openai.ChatOpenAI", DummyModel)

        model = model_initialization.get_llm_model(
            "simple_model",
            vendor="minimax",
            config_path=str(config_path),
            temperature=0.0,
        )

        assert model.kwargs["temperature"] == 0.01

    def test_get_llm_model_clamps_negative_temperature(self, monkeypatch, tmp_path):
        """Negative temperature should also be clamped to 0.01."""
        config_path = write_config(tmp_path / "config.toml", MINIMAX_CONFIG_TEMPLATE)
        monkeypatch.setattr("langchain_openai.ChatOpenAI", DummyModel)

        model = model_initialization.get_llm_model(
            "simple_model",
            vendor="minimax",
            config_path=str(config_path),
            temperature=-0.5,
        )

        assert model.kwargs["temperature"] == 0.01

    def test_get_llm_model_positive_temperature_unchanged(self, monkeypatch, tmp_path):
        config_path = write_config(tmp_path / "config.toml", MINIMAX_CONFIG_TEMPLATE)
        monkeypatch.setattr("langchain_openai.ChatOpenAI", DummyModel)

        model = model_initialization.get_llm_model(
            "simple_model",
            vendor="minimax",
            config_path=str(config_path),
            temperature=0.5,
        )

        assert model.kwargs["temperature"] == 0.5


# ---------------------------------------------------------------------------
# get_llm_model_direct factory
# ---------------------------------------------------------------------------


class TestMiniMaxLLMModelDirect:
    """Tests for get_llm_model_direct with MiniMax vendor."""

    def test_get_llm_model_direct_uses_provided_model_name(self, monkeypatch, tmp_path):
        config_path = write_config(tmp_path / "config.toml", MINIMAX_CONFIG_TEMPLATE)
        monkeypatch.setattr("langchain_openai.ChatOpenAI", DummyModel)

        model = model_initialization.get_llm_model_direct(
            model_name="MiniMax-M2.7-highspeed",
            vendor="minimax",
            config_path=str(config_path),
        )

        assert isinstance(model, DummyModel)
        assert model.kwargs["model"] == "MiniMax-M2.7-highspeed"
        assert model.kwargs["base_url"] == "https://api.minimax.io/v1"

    def test_get_llm_model_direct_clamps_zero_temperature(self, monkeypatch, tmp_path):
        config_path = write_config(tmp_path / "config.toml", MINIMAX_CONFIG_TEMPLATE)
        monkeypatch.setattr("langchain_openai.ChatOpenAI", DummyModel)

        model = model_initialization.get_llm_model_direct(
            model_name="MiniMax-M2.7",
            vendor="minimax",
            config_path=str(config_path),
            temperature=0.0,
        )

        assert model.kwargs["temperature"] == 0.01

    def test_get_llm_model_direct_reads_api_key(self, monkeypatch, tmp_path):
        config_path = write_config(tmp_path / "config.toml", MINIMAX_CONFIG_TEMPLATE)
        monkeypatch.setenv("MINIMAX_API_KEY", "direct-key")
        monkeypatch.setattr("langchain_openai.ChatOpenAI", DummyModel)

        model = model_initialization.get_llm_model_direct(
            model_name="MiniMax-M2.7",
            vendor="minimax",
            config_path=str(config_path),
        )

        assert model.kwargs["api_key"] == "direct-key"


# ---------------------------------------------------------------------------
# Embeddings — MiniMax not supported
# ---------------------------------------------------------------------------


class TestMiniMaxEmbeddingsNotSupported:
    """MiniMax does not expose a public embeddings API."""

    def test_get_embeddings_model_raises_for_minimax(self, tmp_path):
        minimax_embeddings_config = MINIMAX_CONFIG_TEMPLATE.replace(
            'embeddings_model = "openai"', 'embeddings_model = "minimax"', 1
        )
        config_path = write_config(tmp_path / "config.toml", minimax_embeddings_config)

        with pytest.raises(
            ValueError, match="MiniMax does not provide a public embeddings API"
        ):
            model_initialization.get_embeddings_model(config_path=str(config_path))
