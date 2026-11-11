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

"""RAI settings: `RAI_*` env vars over an optional `rai.toml` over defaults.

The file is the first of: an explicit path, `$RAI_CONFIG`, `./rai.toml`.
Nested env vars use `__`, e.g. `RAI_MODELS__COMPLEX=ollama:llama3.1:70b`.
Secrets never go in the file; providers read their own env vars
(`OPENAI_API_KEY`, `LANGFUSE_SECRET_KEY`, ...).
"""

import functools
import os
import tomllib
from contextvars import ContextVar
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

M = TypeVar("M", bound=BaseModel)

_config_file: ContextVar[Path | None] = ContextVar("rai_config_file", default=None)


class ModelSettings(BaseModel):
    """Models as `"provider:model"` strings, the `init_chat_model` syntax."""

    model_config = ConfigDict(extra="forbid")

    simple: str = "openai:gpt-4o-mini"
    complex: str = "openai:gpt-4o"
    embeddings: str = "openai:text-embedding-3-small"
    # extra roles, e.g. {"vlm": "google_genai:gemini-3-flash"}
    aliases: dict[str, str] = {}
    # per-provider constructor kwargs, e.g. {"openai": {"base_url": "http://localhost:8000/v1"}}
    provider_kwargs: dict[str, dict[str, Any]] = {}


class TracingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str = "rai"
    langfuse: bool = False  # keys in LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
    langfuse_host: str | None = None
    langsmith: bool = False  # key in LANGSMITH_API_KEY
    langsmith_host: str | None = None


class RAISettings(BaseSettings):
    # Unknown top-level tables are ignored, so other packages can keep their own
    # sections (e.g. rai_s2s's [asr] and [tts]) in the same file; see `section`.
    model_config = SettingsConfigDict(
        env_prefix="RAI_", env_nested_delimiter="__", extra="ignore"
    )

    models: ModelSettings = ModelSettings()
    tracing: TracingSettings = TracingSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml_file = _config_file.get() or find_config_file()
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=toml_file),
        )


def find_config_file(path: str | os.PathLike[str] | None = None) -> Path | None:
    """Return the settings file: `path`, else `$RAI_CONFIG`, else `./rai.toml` if present."""
    explicit = path or os.environ.get("RAI_CONFIG")
    if explicit:
        if not Path(explicit).is_file():
            raise FileNotFoundError(f"RAI settings file not found: {explicit}")
        return Path(explicit)
    local = Path("rai.toml")
    return local if local.is_file() else None


def load_settings(path: str | os.PathLike[str] | None = None) -> RAISettings:
    """Build settings from scratch (no cache)."""
    token = _config_file.set(find_config_file(path))
    try:
        return RAISettings()
    finally:
        _config_file.reset(token)


@functools.cache
def get_settings(path: str | os.PathLike[str] | None = None) -> RAISettings:
    """Cached `load_settings`. Call `get_settings.cache_clear()` after changing env or files."""
    return load_settings(path)


def section(name: str, model: type[M], path: str | os.PathLike[str] | None = None) -> M:
    """Read the `[name]` table of the settings file into `model`.

    `RAI_<NAME>__<FIELD>` env vars override the file. For packages outside
    rai_core (e.g. `section("asr", ASRSettings)`), so core needs no knowledge of them.
    """
    file = find_config_file(path)
    data: dict[str, Any] = {}
    if file is not None:
        with file.open("rb") as f:
            data = dict(tomllib.load(f).get(name, {}))
    prefix = f"RAI_{name.upper()}__"
    data |= {
        k[len(prefix) :].lower(): v
        for k, v in os.environ.items()
        if k.upper().startswith(prefix)
    }
    return model.model_validate(data)
