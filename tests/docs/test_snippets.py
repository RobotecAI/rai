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

"""The Python and TOML snippets of the docs must match the code.

Every ```python block of the checked pages is compiled, every name it imports
is imported, and attribute chains on imported modules (``rai.run``) are
resolved. Blocks preceded by ``<!-- test: run -->`` are executed, in page
order, in one namespace per page: they must need no ROS graph and no API key
(the chat model is a fake). ``<!-- test: file NAME -->`` writes the next block
to NAME in the page's working directory before the page runs, and
``<!-- test: skip -->`` leaves a block out. Every ```toml block must parse.
"""

import ast
import asyncio
import importlib
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

DOCS = Path(__file__).resolve().parents[2] / "docs"
CHECKED = [
    "index.md",
    "getting_started",
    "concepts",
    "guides",
    "speech_to_speech",
    "simulation_and_benchmarking",
    "demos",
]
# Words of the 2.x API that must not come back (not checked on the demo pages,
# which are left as they are).
OBSOLETE = [
    "config.toml",
    "get_llm_model",
    "rai-config-init",
    "AgentRunner",
    "BaseConnector",
    "MultimodalMessage",
    "ToolRunner",
    "create_react_runnable",
    "langchain_community",
    "Humble",
    "humble",
]

FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```", re.M | re.S)
MARKER = re.compile(r"<!--\s*test:\s*(\w+)\s*([^>]*?)\s*-->\s*$")


@dataclass
class Block:
    page: str
    index: int
    lang: str
    code: str
    marker: str = ""
    arg: str = ""

    @property
    def id(self) -> str:
        return f"{self.page}:{self.index}"


def pages() -> list[Path]:
    found: list[Path] = []
    for entry in CHECKED:
        path = DOCS / entry
        found += sorted(path.rglob("*.md")) if path.is_dir() else [path]
    return found


def blocks_of(path: Path) -> list[Block]:
    text = path.read_text()
    out = []
    for i, match in enumerate(FENCE.finditer(text)):
        before = text[: match.start()].rstrip().splitlines()
        marker = MARKER.match(before[-1].strip()) if before else None
        out.append(
            Block(
                str(path.relative_to(DOCS)),
                i,
                match.group(1),
                match.group(2),
                marker.group(1) if marker else "",
                marker.group(2) if marker else "",
            )
        )
    return out


ALL = [b for p in pages() for b in blocks_of(p)]
PYTHON = [b for b in ALL if b.lang == "python" and b.marker != "skip"]
TOML = [b for b in ALL if b.lang == "toml"]
RUN_PAGES = sorted({b.page for b in PYTHON if b.marker == "run"})


def compiled(block: Block) -> ast.Module:
    return compile(
        block.code,
        block.id,
        "exec",
        flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        dont_inherit=True,
    )


def resolve(module: str, attrs: list[str]) -> None:
    """Import module, then each attribute in turn (importing submodules)."""
    obj: Any = importlib.import_module(module)
    path = module
    for attr in attrs:
        path = f"{path}.{attr}"
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            obj = importlib.import_module(path)


def attribute_chain(node: ast.Attribute) -> tuple[str, list[str]] | None:
    attrs = []
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value  # type: ignore[assignment]
    return (node.id, attrs[::-1]) if isinstance(node, ast.Name) else None


@pytest.mark.parametrize("block", PYTHON, ids=lambda b: b.id)
def test_python_snippet_compiles_and_imports(block: Block):
    tree = compiled(block)
    modules: dict[str, str] = {}  # local name -> module
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolve(alias.name, [])
                local = alias.asname or alias.name.split(".")[0]
                modules[local] = alias.name if alias.asname else local
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                resolve(node.module, [alias.name])
                try:
                    importlib.import_module(f"{node.module}.{alias.name}")
                    modules[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                except ImportError:
                    pass
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            chain = attribute_chain(node)
            if chain and chain[0] in modules:
                check_attributes(modules[chain[0]], chain[1])


def check_attributes(module: str, attrs: list[str]) -> None:
    """Resolve attrs on module down to the first object that is not a module;
    attributes of that object are runtime values and are not checked."""
    obj: Any = importlib.import_module(module)
    for attr in attrs:
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            obj = importlib.import_module(f"{obj.__name__}.{attr}")
        if not isinstance(obj, ModuleType):
            return


@pytest.mark.parametrize("block", TOML, ids=lambda b: b.id)
def test_toml_snippet_parses(block: Block):
    tomllib.loads(block.code)


class FakeChatModel(BaseChatModel):
    """Answers every request with a fixed text, so run blocks need no API key."""

    @property
    def _llm_type(self) -> str:
        return "docs-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeChatModel":
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage("Done."))])


@pytest.fixture
def docs_env(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("RAI_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    import langchain.chat_models
    import rai.settings
    from rai.tools import global_tool_hooks

    def fake_chat_model(*args: Any, **kwargs: Any) -> FakeChatModel:
        return FakeChatModel()

    # rai.llm.get_llm imports init_chat_model on each call
    monkeypatch.setattr(langchain.chat_models, "init_chat_model", fake_chat_model)
    rai.settings.get_settings.cache_clear()
    yield tmp_path
    rai.settings.get_settings.cache_clear()
    global_tool_hooks.clear()


@pytest.mark.parametrize("page", RUN_PAGES)
def test_run_snippets(page: str, docs_env: Path):
    blocks = blocks_of(DOCS / page)
    for block in blocks:
        if block.marker == "file":
            (docs_env / block.arg).write_text(block.code)
    namespace: dict[str, Any] = {"__name__": f"docs_{Path(page).stem}"}
    for block in blocks:
        if block.marker != "run":
            continue
        code = compile(block.code, block.id, "exec", dont_inherit=True)
        exec(code, namespace)
    asyncio.set_event_loop(None)


@pytest.mark.parametrize(
    "path",
    [p for p in pages() if p.relative_to(DOCS).parts[0] != "demos"],
    ids=lambda p: str(p.relative_to(DOCS)),
)
def test_no_obsolete_api(path: Path):
    text = path.read_text()
    found = [word for word in OBSOLETE if word in text]
    assert not found, f"{path.relative_to(DOCS)} mentions the 2.x API: {found}"
