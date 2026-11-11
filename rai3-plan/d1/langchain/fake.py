"""Scripted fake chat model (test support, not counted as slice code)."""

from __future__ import annotations

import itertools
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ..common.script import Script, Text


class ScriptedChatModel(BaseChatModel):
    script: Script
    ids: Any = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kw):
        self.ids = self.ids or itertools.count()
        step = self.script.next([(m.type, m.text) for m in messages])
        if isinstance(step, Text):
            msg = AIMessage(step.text)
        else:
            calls = [
                {
                    "name": c.name,
                    "args": c.args,
                    "id": f"call_{next(self.ids)}",
                    "type": "tool_call",
                }
                for c in step
            ]
            msg = AIMessage("", tool_calls=calls)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kw):
        return self._generate(messages, stop, None, **kw)
