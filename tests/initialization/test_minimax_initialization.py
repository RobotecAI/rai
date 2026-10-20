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
from pathlib import Path

import httpx
import pytest
import tomli
import tomli_w
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from rai.initialization import config_initialization, model_initialization

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
simple_model = "MiniMax-M3.1-Flash"
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

    assert config.minimax.simple_model == "MiniMax-M3.1-Flash"
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
    assert model.kwargs["model"] == "MiniMax-M3.1-Flash"
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


@pytest.fixture
def captured_requests(monkeypatch):
    requests = []
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")

    def send(client, request, **kwargs):
        payload = json.loads(request.content)
        requests.append((request, payload))
        if request.url.path.endswith("/messages"):
            response = {
                "id": "message-test",
                "type": "message",
                "role": "assistant",
                "model": payload["model"],
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        else:
            response = {
                "id": "chat-test",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        return httpx.Response(200, json=response, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    return requests


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "region,host", [("global_en", "api.minimax.io"), ("cn_zh", "api.minimaxi.com")]
)
@pytest.mark.parametrize("source", ["generated", "repository"])
@pytest.mark.parametrize(
    "model_type,expected",
    [("simple_model", "MiniMax-M3.1-Flash"), ("complex_model", "MiniMax-M3")],
)
def test_default_request_payload(
    tmp_path,
    monkeypatch,
    captured_requests,
    protocol,
    region,
    host,
    source,
    model_type,
    expected,
):
    if source == "generated":
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["rai-config-init"])
        config_initialization.main()
        config_path = tmp_path / "config.toml"
    else:
        config_path = Path(__file__).resolve().parents[2] / "config.toml"
    config = tomli.loads(config_path.read_text())
    config["vendor"][model_type] = "minimax"
    config["minimax"].update(protocol=protocol, region=region)
    config_path = write_config(tmp_path / "request.toml", tomli_w.dumps(config))

    model = model_initialization.get_llm_model(model_type, config_path=str(config_path))
    assert model.invoke("Say hello").content == "ok"

    request, payload = captured_requests.pop()
    assert request.url.host == host
    assert request.url.path == (
        "/v1/chat/completions" if protocol == "openai" else "/anthropic/v1/messages"
    )
    assert payload["model"] == expected
    assert payload["messages"][0]["content"] == "Say hello"
    assert "service_tier" not in payload
    assert "reasoning_split" not in payload
    assert payload.get("thinking", {}).get("type") != "disabled"
    assert payload.get("reasoning_effort") != "none"
    assert payload.get("output_config", {}).get("effort") != "none"


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("model_name", ["MiniMax-M2.7", "custom-deployment"])
@pytest.mark.parametrize("direct", [False, True])
def test_saved_and_explicit_models_survive_request(
    tmp_path, monkeypatch, captured_requests, protocol, model_name, direct
):
    config = tomli.loads(MINIMAX_CONFIG_TEMPLATE)
    config["minimax"]["protocol"] = protocol
    config["minimax"]["simple_model"] = model_name
    if model_name == "custom-deployment":
        config["minimax"]["endpoints"]["global_en"] = {
            "openai_base_url": "https://custom.example/v1",
            "anthropic_base_url": "https://custom.example/anthropic",
        }
    config_path = write_config(tmp_path / "config.toml", tomli_w.dumps(config))
    saved = config_path.read_bytes()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["rai-config-init"])
    config_initialization.main()
    assert config_path.read_bytes() == saved
    if direct:
        model = model_initialization.get_llm_model_direct(
            model_name, "minimax", str(config_path)
        )
    else:
        model = model_initialization.get_llm_model(
            "simple_model", config_path=str(config_path)
        )
    model.invoke("Say hello")
    request, payload = captured_requests.pop()
    assert payload["model"] == model_name
    assert request.url.host == (
        "custom.example" if model_name == "custom-deployment" else "api.minimax.io"
    )


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("model_name", ["MiniMax-M3.1-Flash", "MiniMax-M3"])
def test_request_preserves_options_tools_and_images(
    tmp_path, captured_requests, protocol, model_name
):
    config = tomli.loads(MINIMAX_CONFIG_TEMPLATE)
    config["minimax"]["protocol"] = protocol
    config_path = write_config(tmp_path / "config.toml", tomli_w.dumps(config))
    kwargs = (
        {"reasoning_effort": "high"}
        if protocol == "openai"
        else {
            "thinking": {"type": "adaptive"},
            "model_kwargs": {"output_config": {"effort": "high"}},
        }
    )
    model = model_initialization.get_llm_model_direct(
        model_name, "minimax", str(config_path), api_key="explicit-key", **kwargs
    )
    messages = [
        SystemMessage(content="Describe the scene."),
        HumanMessage(content="Look up the cup."),
        AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "lookup", "args": {"label": "cup"}}],
        ),
        ToolMessage(content="A blue cup", tool_call_id="call-1"),
        HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                },
            ]
        ),
    ]
    tool = {
        "name": "lookup",
        "description": "Look up a label",
        "parameters": {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        },
    }
    model.bind_tools([tool]).invoke(messages)
    request, payload = captured_requests.pop()
    assert payload["model"] == model_name
    if protocol == "openai":
        assert request.headers["authorization"] == "Bearer explicit-key"
        assert payload["reasoning_effort"] == "high"
        assert payload["messages"][0]["content"] == "Describe the scene."
        assert payload["messages"][2]["tool_calls"][0]["id"] == "call-1"
        assert payload["messages"][3]["tool_call_id"] == "call-1"
        assert (
            payload["messages"][-1]["content"][-1]["image_url"]["url"]
            == "data:image/png;base64,aW1hZ2U="
        )
        assert payload["tools"][0]["function"]["name"] == "lookup"
    else:
        assert request.headers["x-api-key"] == "explicit-key"
        assert payload["thinking"] == {"type": "adaptive"}
        assert payload["output_config"] == {"effort": "high"}
        assert payload["system"] == "Describe the scene."
        assert payload["messages"][1]["content"][0]["id"] == "call-1"
        assert payload["messages"][-1]["content"][0]["tool_use_id"] == "call-1"
        assert payload["messages"][-1]["content"][-1]["source"]["data"] == "aW1hZ2U="
        assert payload["tools"][0]["name"] == "lookup"
