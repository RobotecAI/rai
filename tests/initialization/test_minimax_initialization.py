# Copyright (C) 2026 Robotec.AI
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

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from rai.initialization import model_initialization

MINIMAX_CONFIG_TEMPLATE = """
[vendor]
simple_model = "minimax"
complex_model = "minimax"
embeddings_model = "openai"

[openai]
simple_model = "gpt-4o-mini"
complex_model = "gpt-4o"
embeddings_model = "text-embedding-3-small"
base_url = "https://api.openai.com/v1"

[minimax]
simple_model = "MiniMax-M2.7"
complex_model = "MiniMax-M3"
embeddings_model = ""
protocol = "openai"
region = "global_en"

[minimax.endpoints.global_en]
openai_base_url = "https://api.minimax.io/v1"
anthropic_base_url = "https://api.minimax.io/anthropic"

[minimax.endpoints.cn_zh]
openai_base_url = "https://api.minimaxi.com/v1"
anthropic_base_url = "https://api.minimaxi.com/anthropic"
"""


class DummyModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def write_config(path: Path, config: str = MINIMAX_CONFIG_TEMPLATE) -> Path:
    path.write_text(config, encoding="utf-8")
    return path


def test_load_config_preserves_models_and_endpoints(tmp_path):
    config = model_initialization.load_config(
        str(write_config(tmp_path / "config.toml"))
    )

    assert config.minimax.simple_model == "MiniMax-M2.7"
    assert config.minimax.complex_model == "MiniMax-M3"
    assert config.minimax.endpoints["global_en"].openai_base_url.endswith("/v1")
    assert config.minimax.endpoints["global_en"].anthropic_base_url.endswith(
        "/anthropic"
    )
    assert config.minimax.endpoints["cn_zh"].openai_base_url.endswith("/v1")
    assert config.minimax.endpoints["cn_zh"].anthropic_base_url.endswith("/anthropic")


def test_get_llm_model_uses_openai_protocol_and_selected_region(monkeypatch, tmp_path):
    config_path = write_config(tmp_path / "config.toml")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr("langchain_openai.ChatOpenAI", DummyModel)

    model = model_initialization.get_llm_model(
        "complex_model", vendor="minimax", config_path=str(config_path)
    )

    assert isinstance(model, DummyModel)
    assert model.kwargs["model"] == "MiniMax-M3"
    assert model.kwargs["base_url"] == "https://api.minimax.io/v1"
    assert model.kwargs["api_key"] == "test-key"


def test_get_llm_model_uses_anthropic_protocol(monkeypatch, tmp_path):
    config = MINIMAX_CONFIG_TEMPLATE.replace(
        'protocol = "openai"', 'protocol = "anthropic"'
    )
    config_path = write_config(tmp_path / "config.toml", config)
    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", DummyModel)

    model = model_initialization.get_llm_model(
        "simple_model", vendor="minimax", config_path=str(config_path)
    )

    assert isinstance(model, DummyModel)
    assert model.kwargs["model"] == "MiniMax-M2.7"
    assert model.kwargs["base_url"] == "https://api.minimax.io/anthropic"


def test_get_llm_model_direct_preserves_explicit_api_key(monkeypatch, tmp_path):
    config_path = write_config(tmp_path / "config.toml")
    monkeypatch.setenv("MINIMAX_API_KEY", "environment-key")
    monkeypatch.setattr("langchain_openai.ChatOpenAI", DummyModel)

    model = model_initialization.get_llm_model_direct(
        "MiniMax-M3",
        vendor="minimax",
        config_path=str(config_path),
        api_key="explicit-key",
    )

    assert model.kwargs["api_key"] == "explicit-key"


def test_get_embeddings_model_rejects_minimax(tmp_path):
    config = MINIMAX_CONFIG_TEMPLATE.replace(
        'embeddings_model = "openai"', 'embeddings_model = "minimax"'
    )
    config_path = write_config(tmp_path / "config.toml", config)

    with pytest.raises(ValueError, match="does not provide an embeddings model"):
        model_initialization.get_embeddings_model(config_path=str(config_path))


class CaptureHandler(BaseHTTPRequestHandler):
    request_path = ""

    def do_POST(self):  # noqa: N802
        self.__class__.request_path = self.path
        content_length = int(self.headers.get("content-length", "0"))
        self.rfile.read(content_length)
        response = {
            "id": "message-test",
            "type": "message",
            "role": "assistant",
            "model": "MiniMax-M3",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):  # noqa: A002
        return


def test_anthropic_client_appends_messages_path(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/anthropic"
        config = MINIMAX_CONFIG_TEMPLATE.replace(
            'protocol = "openai"', 'protocol = "anthropic"'
        ).replace("https://api.minimax.io/anthropic", base_url)
        config_path = write_config(tmp_path / "config.toml", config)
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")

        model = model_initialization.get_llm_model(
            "complex_model", vendor="minimax", config_path=str(config_path)
        )
        response = model.invoke("Say hello")

        assert response.content == "ok"
        assert CaptureHandler.request_path == "/anthropic/v1/messages"
    finally:
        server.shutdown()
        thread.join(timeout=5)
