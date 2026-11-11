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

import os

import pytest
from rai import llm
from rai.llm import get_tracing_callbacks
from rai.settings import get_settings


@pytest.fixture(autouse=True)
def clean_settings(tmp_path, monkeypatch):
    """No RAI_* env, an empty CWD and fresh caches, so only the test's inputs count."""
    for key in list(os.environ):
        if key.startswith("RAI_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    get_tracing_callbacks.cache_clear()
    yield
    get_settings.cache_clear()
    get_tracing_callbacks.cache_clear()


@pytest.fixture
def calls(monkeypatch):
    """Record what reaches LangChain instead of building real clients."""
    import langchain.chat_models
    import langchain.embeddings

    seen = []
    monkeypatch.setattr(
        langchain.chat_models,
        "init_chat_model",
        lambda model, **kw: seen.append((model, kw)) or "chat",
    )
    monkeypatch.setattr(
        langchain.embeddings,
        "init_embeddings",
        lambda model, **kw: seen.append((model, kw)) or "emb",
    )
    monkeypatch.setattr(llm, "_require_provider", lambda provider: None)
    return seen
