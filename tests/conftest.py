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
import tempfile

import pytest


# 3D gripping point detection strategy
def pytest_addoption(parser):
    parser.addoption(
        "--strategy", action="store", default="centroid", help="Gripping point strategy"
    )


@pytest.fixture
def strategy(request):
    return request.config.getoption("--strategy")


@pytest.fixture
def test_config_toml():
    """
    Fixture to create a temporary test config.toml file with tracing enabled.

    Returns
    -------
    tuple
        (config_path, cleanup_function) - The path to the config file and a function to clean it up
    """

    def _create_config(langfuse_enabled=False, langsmith_enabled=False):
        # Create a temporary config.toml file
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)

        # Base config sections (always required)
        config_content = """[vendor]
simple_model = "openai"
complex_model = "openai"
embeddings_model = "text-embedding-ada-002"

[aws]
simple_model = "anthropic.claude-instant-v1"
complex_model = "anthropic.claude-v2"
embeddings_model = "amazon.titan-embed-text-v1"
region_name = "us-east-1"

[openai]
simple_model = "gpt-3.5-turbo"
complex_model = "gpt-4"
embeddings_model = "text-embedding-ada-002"
base_url = "https://api.openai.com/v1"

[ollama]
simple_model = "llama2"
complex_model = "llama2"
embeddings_model = "llama2"
base_url = "http://localhost:11434"

[tracing]
project = "test-project"

[tracing.langfuse]
use_langfuse = {langfuse_enabled}
host = "http://localhost:3000"

[tracing.langsmith]
use_langsmith = {langsmith_enabled}
host = "https://api.smith.langchain.com"
""".format(
            langfuse_enabled=str(langfuse_enabled).lower(),
            langsmith_enabled=str(langsmith_enabled).lower(),
        )

        f.write(config_content)
        f.close()

        def cleanup():
            try:
                f.close()  # Ensure file is properly closed
                os.unlink(f.name)
            except (OSError, PermissionError):
                pass  # File might already be deleted or have permission issues

        return f.name, cleanup

    return _create_config


@pytest.fixture
def test_config_no_tracing():
    """
    Fixture to create a temporary test config.toml file with no tracing section.

    Returns
    -------
    tuple
        (config_path, cleanup_function) - The path to the config file and a function to clean it up
    """

    def _create_config():
        # Create a temporary config.toml file
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)

        # Base config sections (always required)
        config_content = """[vendor]
simple_model = "openai"
complex_model = "openai"
embeddings_model = "text-embedding-ada-002"

[aws]
simple_model = "anthropic.claude-instant-v1"
complex_model = "anthropic.claude-v2"
embeddings_model = "amazon.titan-embed-text-v1"
region_name = "us-east-1"

[openai]
simple_model = "gpt-3.5-turbo"
complex_model = "gpt-4"
embeddings_model = "text-embedding-ada-002"
base_url = "https://api.openai.com/v1"

[ollama]
simple_model = "llama2"
complex_model = "llama2"
embeddings_model = "llama2"
base_url = "http://localhost:11434"
"""

        f.write(config_content)
        f.close()

        def cleanup():
            try:
                f.close()  # Ensure file is properly closed
                os.unlink(f.name)
            except (OSError, PermissionError):
                pass  # File might already be deleted or have permission issues

        return f.name, cleanup

    return _create_config
