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

"""Render LangGraph agent runs (streamed text, tool calls, images) in Streamlit."""

import base64
import json
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, List
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx


class StreamlitCallbackHandler(BaseCallbackHandler):
    """Streams LLM tokens and shows one status block per tool call inside a container."""

    def __init__(self, parent_container: DeltaGenerator):
        self._ctx = get_script_run_ctx()
        self._lock = threading.Lock()
        self._tools = parent_container.container()
        self._answer = parent_container.empty()
        self._text = ""
        self._calls: dict[UUID, tuple[Any, str, float]] = {}

    @contextmanager
    def _ui(self) -> Iterator[None]:
        """Serialize UI writes and bind the calling thread to the script run context."""
        with self._lock:
            add_script_run_ctx(threading.current_thread(), self._ctx)
            yield

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    ) -> None:
        pass

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if not token:
            return
        with self._ui():
            self._text += token
            self._answer.markdown(self._text + " ▌")

    def on_llm_end(self, response: Any, **kwargs) -> None:
        if not self._text:
            return
        with self._ui():
            self._answer.markdown(self._text)
            if not self._text.endswith("\n\n"):
                self._text += "\n\n"

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        serialized = serialized or {}
        name = serialized.get("name", "tool")
        with self._ui():
            status = self._tools.status(f"**{name}**", state="running")
            if serialized.get("description"):
                status.caption(serialized["description"])
            args = input_str if inputs is None else json.dumps(inputs, default=str)
            status.code(args, language="json", wrap_lines=True)
            self._calls[run_id] = (status, name, time.monotonic())

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs) -> None:
        with self._ui():
            call = self._calls.pop(run_id, None)
            if call is None:
                return
            status, name, started = call
            content = getattr(output, "content", output)
            artifact = getattr(output, "artifact", None)
            images = artifact.get("images", []) if isinstance(artifact, dict) else []
            if content:
                status.code(str(content), language=None, wrap_lines=True)
            for image in images:
                status.image(base64.b64decode(image), width="stretch")
            status.update(
                label=f"**{name}** :gray[{time.monotonic() - started:.1f}s]",
                state="complete",
                expanded=bool(images),
            )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs) -> None:
        with self._ui():
            call = self._calls.pop(run_id, None)
            if call is None:
                return
            status, name, _ = call
            status.error(f"{type(error).__name__}: {error}")
            status.update(
                label=f"**{name}** :red[failed]", state="error", expanded=True
            )


def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    """Create a callback handler that renders into `parent_container`."""
    return StreamlitCallbackHandler(parent_container)


def streamlit_invoke(
    graph: Runnable, messages: List[BaseMessage], callables: List[BaseCallbackHandler]
) -> Any:
    """Invoke `graph` on `messages` with the given callback handlers. `messages`
    is updated in place with the new messages of the run."""
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    result = graph.invoke(
        {"messages": messages},
        config=RunnableConfig(callbacks=callables, recursion_limit=100),
    )
    messages[:] = result["messages"]
    return result
