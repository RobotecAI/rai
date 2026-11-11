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

import importlib.util

import pytest
from rai.llm import get_embeddings, get_llm, get_tracing_callbacks, parse_model
from rai.settings import ModelSettings, RAISettings


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("openai:gpt-4o", ("openai", "gpt-4o")),
        ("ollama:llama3.1:70b", ("ollama", "llama3.1:70b")),
        ("google:gemini-3-pro", ("google_genai", "gemini-3-pro")),
        ("aws:anthropic.claude-v2", ("bedrock_converse", "anthropic.claude-v2")),
    ],
)
def test_parse_model(spec, expected):
    assert parse_model(spec) == expected


@pytest.mark.parametrize("spec", ["gpt-4o", ":gpt-4o", "openai:"])
def test_parse_model_rejects_bare_names(spec):
    with pytest.raises(ValueError, match="provider:model"):
        parse_model(spec)


SETTINGS = RAISettings(
    models=ModelSettings(
        simple="ollama:llama3.2",
        complex="openai:gpt-4o",
        embeddings="ollama:nomic-embed-text",
        aliases={"vlm": "google:gemini-3-flash"},
        provider_kwargs={
            "openai": {"base_url": "http://vllm:8000/v1", "temperature": 0.5},
            "ollama": {"base_url": "http://robot:11434"},
        },
    )
)


def test_roles_aliases_and_specs_resolve(calls):
    get_llm("simple", settings=SETTINGS)
    get_llm(settings=SETTINGS, temperature=0.0, streaming=True)
    get_llm("vlm", settings=SETTINGS)
    get_llm("anthropic:claude-sonnet-4-5", settings=SETTINGS)
    get_embeddings(settings=SETTINGS)
    assert calls == [
        ("llama3.2", {"model_provider": "ollama", "base_url": "http://robot:11434"}),
        (
            "gpt-4o",
            {
                "model_provider": "openai",
                "base_url": "http://vllm:8000/v1",
                "temperature": 0.0,
                "streaming": True,
            },
        ),
        ("gemini-3-flash", {"model_provider": "google_genai"}),
        ("claude-sonnet-4-5", {"model_provider": "anthropic"}),
        (
            "nomic-embed-text",
            {"provider": "ollama", "base_url": "http://robot:11434"},
        ),
    ]


def test_defaults_come_from_get_settings(calls, monkeypatch):
    monkeypatch.setenv("RAI_MODELS__COMPLEX", "ollama:qwen3")
    get_llm()
    assert calls == [("qwen3", {"model_provider": "ollama"})]


def test_missing_extra_names_it(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "langchain_anthropic" else real(name, *a),
    )
    with pytest.raises(ImportError, match=r"rai_core\[anthropic\]"):
        get_llm("anthropic:claude-sonnet-4-5")


def test_real_openai_compatible_model_gets_base_url():
    pytest.importorskip("langchain_openai")
    settings = RAISettings(
        models=ModelSettings(
            provider_kwargs={"openai": {"base_url": "http://localhost:8000/v1"}}
        )
    )
    model = get_llm("openai:my-model", settings=settings, api_key="sk-test")
    assert model.openai_api_base == "http://localhost:8000/v1"
    assert model.model_name == "my-model"


def test_tracing_off_by_default():
    assert get_tracing_callbacks() == []


def test_langfuse_tracing_is_built_once(monkeypatch):
    pytest.importorskip("langfuse")
    monkeypatch.setenv("RAI_TRACING__LANGFUSE", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    callbacks = get_tracing_callbacks()
    assert len(callbacks) == 1
    assert get_tracing_callbacks() is callbacks


def test_langfuse_without_keys_fails(monkeypatch):
    pytest.importorskip("langfuse")
    monkeypatch.setenv("RAI_TRACING__LANGFUSE", "true")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="LANGFUSE"):
        get_tracing_callbacks()
