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

import pytest
from pydantic import BaseModel, ValidationError
from rai.settings import RAISettings, get_settings, load_settings, section

FILE = """
[models]
simple = "ollama:llama3.2"
complex = "anthropic:claude-sonnet-4-5"

[models.provider_kwargs.ollama]
base_url = "http://robot:11434"

[tracing]
langfuse = true

[asr]
language = "pl"
"""


def test_defaults_without_file_or_env():
    s = load_settings()
    assert s == RAISettings()
    assert s.models.complex == "openai:gpt-4o"
    assert s.tracing.langfuse is False


def test_env_over_file_over_defaults(tmp_path, monkeypatch):
    (tmp_path / "rai.toml").write_text(FILE)
    monkeypatch.setenv("RAI_MODELS__COMPLEX", "ollama:llama3.1:70b")
    s = load_settings()
    assert s.models.complex == "ollama:llama3.1:70b"  # env
    assert s.models.simple == "ollama:llama3.2"  # file
    assert s.models.embeddings == "openai:text-embedding-3-small"  # default
    assert s.models.provider_kwargs == {"ollama": {"base_url": "http://robot:11434"}}
    assert s.tracing.langfuse is True


def test_nested_env_into_dict(monkeypatch):
    monkeypatch.setenv(
        "RAI_MODELS__PROVIDER_KWARGS__OPENAI__BASE_URL", "http://localhost:8000/v1"
    )
    s = load_settings()
    assert s.models.provider_kwargs["openai"]["base_url"] == "http://localhost:8000/v1"


def test_search_order_explicit_then_env_var_then_cwd(tmp_path, monkeypatch):
    (tmp_path / "rai.toml").write_text('[models]\nsimple = "openai:cwd"')
    (tmp_path / "env.toml").write_text('[models]\nsimple = "openai:env"')
    (tmp_path / "explicit.toml").write_text('[models]\nsimple = "openai:explicit"')
    assert load_settings().models.simple == "openai:cwd"
    monkeypatch.setenv("RAI_CONFIG", str(tmp_path / "env.toml"))
    assert load_settings().models.simple == "openai:env"
    assert load_settings(tmp_path / "explicit.toml").models.simple == "openai:explicit"


def test_missing_explicit_file_raises(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.toml")
    monkeypatch.setenv("RAI_CONFIG", str(tmp_path / "nope.toml"))
    with pytest.raises(FileNotFoundError):
        load_settings()


def test_old_config_toml_is_not_read(tmp_path):
    (tmp_path / "config.toml").write_text('[models]\nsimple = "openai:old"')
    assert load_settings().models.simple == "openai:gpt-4o-mini"


def test_typo_in_known_section_is_rejected(tmp_path):
    (tmp_path / "rai.toml").write_text('[models]\ncomplx = "openai:gpt-4o"')
    with pytest.raises(ValidationError):
        load_settings()


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


class _Asr(BaseModel):
    language: str = "en"
    vad_threshold: float = 0.5


def test_section_env_over_file_over_defaults(tmp_path, monkeypatch):
    (tmp_path / "rai.toml").write_text(FILE)
    assert section("asr", _Asr) == _Asr(language="pl")
    monkeypatch.setenv("RAI_ASR__VAD_THRESHOLD", "0.3")
    assert section("asr", _Asr) == _Asr(language="pl", vad_threshold=0.3)
    assert section("tts", _Asr) == _Asr()
