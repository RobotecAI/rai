# Copyright (C) 2026 Robotec.AI
from langchain_core.messages import AIMessage, ToolMessage

from rai.agents.langchain.core.tool_runner import ToolRunner


def test_tool_runner_unknown_tool_returns_error_message():
    runner = ToolRunner(tools=[])
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "missing_tool", "args": {}, "id": "call_1", "type": "tool_call"}],
    )
    out = runner.invoke({"messages": [ai]})
    msgs = out["messages"]
    assert len(msgs) == 2
    err = msgs[-1]
    assert isinstance(err, ToolMessage)
    assert err.status == "error"
    assert "Unknown tool" in err.content
    assert err.tool_call_id == "call_1"
