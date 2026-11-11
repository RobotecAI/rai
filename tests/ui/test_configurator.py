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

import tomllib
from pathlib import Path

import pytest
import rai.ui.configurator as configurator
from rai.settings import load_settings

TRACING = {"project": "p", "langfuse": True, "langfuse_host": None}


def test_updated_sets_models_and_base_urls_and_keeps_other_tables():
    data = {
        "models": {
            "complex": "openai:gpt-4o",
            "provider_kwargs": {"openai": {"base_url": "x", "timeout": 5}},
        },
        "asr": {"language": "pl"},
    }
    new = configurator.updated(
        data,
        {"simple": "ollama:llama3.2", "complex": "openai:gpt-4o"},
        {"ollama": "http://robot:11434", "openai": ""},
        TRACING,
    )
    assert new["asr"] == {"language": "pl"}
    models = new["models"]
    assert models["simple"] == "ollama:llama3.2"
    assert models["provider_kwargs"] == {
        "openai": {"timeout": 5},
        "ollama": {"base_url": "http://robot:11434"},
    }
    assert new["tracing"]["project"] == "p" and "langfuse_host" not in new["tracing"]


def test_updated_rejects_a_model_without_provider():
    with pytest.raises(ValueError, match="provider:model"):
        configurator.updated({}, {"complex": "gpt-4o"}, {}, TRACING)


def test_app_saves_a_file_the_settings_load(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAI_CONFIG", raising=False)
    (tmp_path / "rai.toml").write_text('[tts]\nvendor = "OpenTTS"\n')
    at = AppTest.from_file(str(Path(configurator.__file__))).run()
    assert not at.exception
    complex_input = next(t for t in at.text_input if t.label == "complex")
    complex_input.set_value("anthropic:claude-sonnet-4-5").run()
    next(b for b in at.button if b.label == "Save").click().run()
    assert not at.exception

    saved = tomllib.loads((tmp_path / "rai.toml").read_text())
    assert saved["tts"] == {"vendor": "OpenTTS"}
    settings = load_settings(tmp_path / "rai.toml")
    assert settings.models.complex == "anthropic:claude-sonnet-4-5"
