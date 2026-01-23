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

import os
from pathlib import Path

import pytest
from rai.initialization import model_initialization

CONFIG_TEMPLATE = """
[vendor]
simple_model = "openai"
complex_model = "aws"
embeddings_model = "ollama"

[aws]
simple_model = "aws.simple"
complex_model = "aws.complex"
embeddings_model = "aws.embed"
region_name = "us-west-2"

[openai]
simple_model = "gpt-4o-mini"
complex_model = "gpt-4o"
embeddings_model = "text-embedding-ada-002"
base_url = "https://openai.example/v1/"

[ollama]
simple_model = "llama-simple"
complex_model = "llama-complex"
embeddings_model = "llama-embed"
base_url = "http://localhost:11434"

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
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def write_config(path: Path, config: str) -> Path:
    path.write_text(config, encoding="utf-8")
    return path


def test_get_llm_model_config_and_vendor_defaults(tmp_path):
    config_path = write_config(tmp_path / "config.toml", CONFIG_TEMPLATE)

    model_config, vendor = model_initialization.get_llm_model_config_and_vendor(
        "simple_model", config_path=str(config_path)
    )

    assert vendor == "openai"
    assert model_config.simple_model == "gpt-4o-mini"


def test_get_llm_model_config_and_vendor_override(tmp_path):
    config_path = write_config(tmp_path / "config.toml", CONFIG_TEMPLATE)

    model_config, vendor = model_initialization.get_llm_model_config_and_vendor(
        "complex_model", vendor="aws", config_path=str(config_path)
    )

    assert vendor == "aws"
    assert model_config.complex_model == "aws.complex"


def test_get_llm_model_returns_vendor_instance(monkeypatch, tmp_path):
    config_path = write_config(tmp_path / "config.toml", CONFIG_TEMPLATE)

    monkeypatch.setattr("langchain_openai.ChatOpenAI", DummyModel)
    monkeypatch.setattr("langchain_aws.ChatBedrock", DummyModel)
    monkeypatch.setattr("langchain_ollama.ChatOllama", DummyModel)

    openai_model = model_initialization.get_llm_model(
        "simple_model", vendor="openai", config_path=str(config_path)
    )
    assert isinstance(openai_model, DummyModel)
    assert openai_model.kwargs["model"] == "gpt-4o-mini"
    assert openai_model.kwargs["base_url"] == "https://openai.example/v1/"

    aws_model = model_initialization.get_llm_model(
        "complex_model", vendor="aws", config_path=str(config_path)
    )
    assert isinstance(aws_model, DummyModel)
    assert aws_model.kwargs["model_id"] == "aws.complex"
    assert aws_model.kwargs["region_name"] == "us-west-2"

    ollama_model = model_initialization.get_llm_model(
        "embeddings_model", vendor="ollama", config_path=str(config_path)
    )
    assert isinstance(ollama_model, DummyModel)
    assert ollama_model.kwargs["model"] == "llama-embed"
    assert ollama_model.kwargs["base_url"] == "http://localhost:11434"


def test_get_llm_model_unknown_vendor_raises(tmp_path, monkeypatch):
    bad_config = CONFIG_TEMPLATE.replace(
        'simple_model = "openai"', 'simple_model = "unsupported"', 1
    )
    config_path = write_config(tmp_path / "config.toml", bad_config)

    with pytest.raises(AttributeError, match="has no attribute 'unsupported'"):
        model_initialization.get_llm_model("simple_model", config_path=str(config_path))


def test_load_config_default_path(tmp_path):
    """Test that load_config() defaults to './config.toml' in current directory."""
    # Create a config.toml in a temporary directory
    _ = write_config(tmp_path / "config.toml", CONFIG_TEMPLATE)

    # Change to that directory to test default path behavior
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Call load_config without arguments - should load from current directory
        config = model_initialization.load_config()

        # Verify config was loaded correctly
        assert config.vendor.simple_model == "openai"
        assert config.vendor.complex_model == "aws"
        assert config.openai.simple_model == "gpt-4o-mini"
        assert config.aws.region_name == "us-west-2"
        assert config.tracing.project == "rai"
        assert config.tracing.langfuse.use_langfuse is False

    finally:
        # Restore original directory
        os.chdir(original_cwd)


def test_get_embeddings_model_openai_with_base_url(monkeypatch, tmp_path):
    """Test that OpenAI embeddings model receives base_url parameter."""
    config_path = write_config(tmp_path / "config.toml", CONFIG_TEMPLATE)

    # Update config to use OpenAI for embeddings
    openai_embeddings_config = CONFIG_TEMPLATE.replace(
        'embeddings_model = "ollama"', 'embeddings_model = "openai"', 1
    )
    config_path = write_config(tmp_path / "config.toml", openai_embeddings_config)

    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", DummyModel)

    embeddings = model_initialization.get_embeddings_model(config_path=str(config_path))

    assert isinstance(embeddings, DummyModel)
    assert embeddings.kwargs["model"] == "text-embedding-ada-002"
    assert embeddings.kwargs["base_url"] == "https://openai.example/v1/"


def test_get_embeddings_model_return_kwargs_openai(monkeypatch, tmp_path):
    """Test that return_kwargs includes base_url for OpenAI embeddings."""
    openai_embeddings_config = CONFIG_TEMPLATE.replace(
        'embeddings_model = "ollama"', 'embeddings_model = "openai"', 1
    )
    config_path = write_config(tmp_path / "config.toml", openai_embeddings_config)

    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", DummyModel)

    embeddings, kwargs = model_initialization.get_embeddings_model(
        config_path=str(config_path), return_kwargs=True
    )

    assert isinstance(embeddings, DummyModel)
    assert kwargs["model"] == "text-embedding-ada-002"
    assert kwargs["base_url"] == "https://openai.example/v1/"
    assert kwargs["vendor"] == "openai"
    assert "class" in kwargs
