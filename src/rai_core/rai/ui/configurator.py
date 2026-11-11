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

"""Edit the ``[models]`` and ``[tracing]`` tables of ``rai.toml`` (see :mod:`rai.settings`).

::

    streamlit run src/rai_core/rai/ui/configurator.py

Other tables in the file (e.g. rai_s2s's ``[asr]`` and ``[tts]``) are kept as they are.
"""

import tomllib
from pathlib import Path
from typing import Any

from rai.llm import get_embeddings, get_llm, parse_model
from rai.settings import ModelSettings, RAISettings, TracingSettings, find_config_file

ROLES = ("simple", "complex", "embeddings")


def read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def updated(
    data: dict[str, Any],
    models: dict[str, str],
    base_urls: dict[str, str],
    tracing: dict[str, Any],
) -> dict[str, Any]:
    """``data`` (a parsed rai.toml) with new model strings, per-provider
    ``base_url`` (empty removes it) and tracing settings. Raises on invalid values."""
    current = ModelSettings.model_validate(data.get("models", {}))
    kwargs = {p: dict(kw) for p, kw in current.provider_kwargs.items()}
    for provider, url in base_urls.items():
        kw = kwargs.setdefault(provider, {})
        kw.pop("base_url", None)
        if url:
            kw["base_url"] = url
    for spec in models.values():
        parse_model(spec)
    new_models = ModelSettings.model_validate(
        current.model_dump()
        | models
        | {"provider_kwargs": {p: kw for p, kw in kwargs.items() if kw}}
    )
    new_tracing = TracingSettings.model_validate(data.get("tracing", {}) | tracing)
    return data | {
        "models": new_models.model_dump(exclude_none=True),
        "tracing": new_tracing.model_dump(exclude_none=True),
    }


def check(data: dict[str, Any]) -> dict[str, str | None]:
    """Ping each model role. Maps the role to its error, or ``None`` if it answered."""
    settings = RAISettings(models=data["models"], tracing=data["tracing"])
    errors: dict[str, str | None] = {}
    for role in ROLES:
        try:
            if role == "embeddings":
                get_embeddings(settings=settings).embed_query("ping")
            else:
                get_llm(role, settings=settings).invoke("Say OK.")
            errors[role] = None
        except Exception as e:
            errors[role] = f"{type(e).__name__}: {e}"
    return errors


def main() -> None:
    import streamlit as st
    import tomli_w

    st.set_page_config(page_title="RAI configurator", page_icon=":robot:")
    st.title("RAI configurator")
    path = Path(st.text_input("Settings file", str(find_config_file() or "rai.toml")))
    data = read(path)
    models = ModelSettings.model_validate(data.get("models", {}))
    tracing = TracingSettings.model_validate(data.get("tracing", {}))

    st.header("Models")
    st.caption(
        '`"provider:model"`, e.g. `openai:gpt-4o`, `anthropic:claude-sonnet-4-5`, '
        "`ollama:llama3.1:70b`. API keys go in the provider's env var "
        "(`OPENAI_API_KEY`, ...), never in this file."
    )
    roles = {r: st.text_input(r, getattr(models, r)) for r in ROLES}
    providers = sorted({m.partition(":")[0] for m in roles.values() if ":" in m})
    st.caption(
        "Endpoint per provider, e.g. a vLLM or remote Ollama server. Empty means the default."
    )
    base_urls = {
        p: st.text_input(
            f"{p} base_url", models.provider_kwargs.get(p, {}).get("base_url", "")
        )
        for p in providers
    }

    st.header("Tracing")
    project = st.text_input("Project", tracing.project)
    langfuse = st.checkbox(
        "Langfuse (keys in `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`)",
        tracing.langfuse,
    )
    langfuse_host = st.text_input("Langfuse host", tracing.langfuse_host or "")
    langsmith = st.checkbox("LangSmith (key in `LANGSMITH_API_KEY`)", tracing.langsmith)
    langsmith_host = st.text_input("LangSmith host", tracing.langsmith_host or "")

    try:
        new = updated(
            data,
            roles,
            base_urls,
            {
                "project": project,
                "langfuse": langfuse,
                "langfuse_host": langfuse_host or None,
                "langsmith": langsmith,
                "langsmith_host": langsmith_host or None,
            },
        )
    except ValueError as e:
        st.error(str(e))
        return
    st.code(tomli_w.dumps(new), language="toml")
    test, save = st.columns(2)
    if test.button("Test models"):
        for role, error in check(new).items():
            if error:
                st.error(f"{role}: {error}")
            else:
                st.success(f"{role}: OK")
    if save.button("Save"):
        path.write_text(tomli_w.dumps(new))
        st.success(f"Saved {path}")


if __name__ == "__main__":
    main()
