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

"""Tests for Google GenAI integration with RAI framework.

These tests verify that the RAI initialization module correctly supports
the Google vendor configuration, following the same patterns used for
OpenAI, AWS, and Ollama integrations.
"""

from pathlib import Path

from rai.initialization import model_initialization

# Config template including Google vendor section
GOOGLE_CONFIG_TEMPLATE = """
[vendor]
simple_model = "google"
complex_model = "google"
embeddings_model = "google"

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
    """Dummy model for testing factory function routing without real API calls."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def write_config(path: Path, config: str) -> Path:
    """Helper to write config content to a file."""
    path.write_text(config, encoding="utf-8")
    return path


class TestGoogleVendorConfiguration:
    """Tests for Google vendor configuration in RAI initialization module."""

    def test_get_llm_model_config_and_vendor_google_simple(self, tmp_path):
        """Test that config correctly loads google vendor for simple_model."""
        config_path = write_config(tmp_path / "config.toml", GOOGLE_CONFIG_TEMPLATE)

        model_config, vendor = model_initialization.get_llm_model_config_and_vendor(
            "simple_model", config_path=str(config_path)
        )

        assert vendor == "google"
        assert model_config.simple_model == "gemini-3-flash"

    def test_get_llm_model_config_and_vendor_google_complex(self, tmp_path):
        """Test that config correctly loads google vendor for complex_model."""
        config_path = write_config(tmp_path / "config.toml", GOOGLE_CONFIG_TEMPLATE)

        model_config, vendor = model_initialization.get_llm_model_config_and_vendor(
            "complex_model", config_path=str(config_path)
        )

        assert vendor == "google"
        assert model_config.complex_model == "gemini-3-pro"

    def test_get_llm_model_config_and_vendor_google_override(self, tmp_path):
        """Test that vendor can be explicitly overridden to google."""
        # Use a config where default vendor is openai, but override to google
        config_with_openai_default = GOOGLE_CONFIG_TEMPLATE.replace(
            'simple_model = "google"', 'simple_model = "openai"', 1
        )
        config_path = write_config(tmp_path / "config.toml", config_with_openai_default)

        model_config, vendor = model_initialization.get_llm_model_config_and_vendor(
            "simple_model", vendor="google", config_path=str(config_path)
        )

        assert vendor == "google"
        assert model_config.simple_model == "gemini-3-flash"


class TestGoogleLLMModelFactory:
    """Tests for get_llm_model factory function with Google vendor."""

    def test_get_llm_model_returns_google_instance(self, monkeypatch, tmp_path):
        """Test that get_llm_model returns ChatGoogleGenerativeAI for google vendor."""
        config_path = write_config(tmp_path / "config.toml", GOOGLE_CONFIG_TEMPLATE)

        monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", DummyModel)

        google_model = model_initialization.get_llm_model(
            "simple_model", vendor="google", config_path=str(config_path)
        )

        assert isinstance(google_model, DummyModel)
        assert google_model.kwargs["model"] == "gemini-3-flash"

    def test_get_llm_model_complex_returns_correct_model(self, monkeypatch, tmp_path):
        """Test that get_llm_model returns correct complex model for google vendor."""
        config_path = write_config(tmp_path / "config.toml", GOOGLE_CONFIG_TEMPLATE)

        monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", DummyModel)

        google_model = model_initialization.get_llm_model(
            "complex_model", vendor="google", config_path=str(config_path)
        )

        assert isinstance(google_model, DummyModel)
        assert google_model.kwargs["model"] == "gemini-3-pro"

    def test_get_llm_model_passes_kwargs_to_google(self, monkeypatch, tmp_path):
        """Test that additional kwargs are passed through to ChatGoogleGenerativeAI."""
        config_path = write_config(tmp_path / "config.toml", GOOGLE_CONFIG_TEMPLATE)

        monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", DummyModel)

        google_model = model_initialization.get_llm_model(
            "simple_model",
            vendor="google",
            config_path=str(config_path),
            temperature=0.7,
            max_output_tokens=1024,
        )

        assert google_model.kwargs["temperature"] == 0.7
        assert google_model.kwargs["max_output_tokens"] == 1024


class TestGoogleLLMModelDirect:
    """Tests for get_llm_model_direct factory function with Google vendor."""

    def test_get_llm_model_direct_returns_google_instance(self, monkeypatch, tmp_path):
        """Test that get_llm_model_direct returns ChatGoogleGenerativeAI."""
        config_path = write_config(tmp_path / "config.toml", GOOGLE_CONFIG_TEMPLATE)

        monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", DummyModel)

        google_model = model_initialization.get_llm_model_direct(
            model_name="gemini-2.0-flash-exp",
            vendor="google",
            config_path=str(config_path),
        )

        assert isinstance(google_model, DummyModel)
        assert google_model.kwargs["model"] == "gemini-2.0-flash-exp"


class TestGoogleEmbeddingsModel:
    """Tests for get_embeddings_model with Google vendor."""

    def test_get_embeddings_model_google(self, monkeypatch, tmp_path):
        """Test that get_embeddings_model returns GoogleGenerativeAIEmbeddings."""
        config_path = write_config(tmp_path / "config.toml", GOOGLE_CONFIG_TEMPLATE)

        monkeypatch.setattr(
            "langchain_google_genai.GoogleGenerativeAIEmbeddings", DummyModel
        )

        embeddings = model_initialization.get_embeddings_model(
            config_path=str(config_path)
        )

        assert isinstance(embeddings, DummyModel)
        assert embeddings.kwargs["model"] == "text-embedding-004"

    def test_get_embeddings_model_google_with_kwargs(self, monkeypatch, tmp_path):
        """Test that get_embeddings_model returns kwargs for google vendor."""
        config_path = write_config(tmp_path / "config.toml", GOOGLE_CONFIG_TEMPLATE)

        monkeypatch.setattr(
            "langchain_google_genai.GoogleGenerativeAIEmbeddings", DummyModel
        )

        embeddings, kwargs = model_initialization.get_embeddings_model(
            config_path=str(config_path), return_kwargs=True
        )

        assert isinstance(embeddings, DummyModel)
        assert kwargs["model"] == "text-embedding-004"
        assert kwargs["vendor"] == "google"
