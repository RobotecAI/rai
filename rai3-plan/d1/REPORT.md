# D1: agent framework bake-off (LangChain 1.x vs PydanticAI)

Status: done, 2026-11-11. **Decision: LangChain 1.x.** D2 is not triggered.

The same slice was built twice, once per framework, and both pass the same 25 tests on a fake robot and on rai_core's `ROS2Connector` against a real ROS 2 node on Lyrical. Every number below is **measured** unless marked **inferred**.

## 1. What was built

```
d1/
  common/              framework-neutral, shared by both slices
    robot.py           RobotIO: FakeRobot, ConnectorRobot (ROS2Connector), RobotSim (ROS node), tool bodies
    guard.py           fail-closed hook core: check(hooks, call, timeout) -> None | denial text
    script.py          deterministic script for the fake chat models
    serve.py           MCP server entry point: python -m d1.common.serve {langchain|pydanticai} {fake|ros} [deny|crash]
    ext_mcp_server.py  external MCP server (official mcp SDK) consumed by both clients
    mcp_e2e.py         stdio end-to-end driver
  langchain/           slice.py, mcp.py (counted), fake.py, test_langchain_slice.py
  pydanticai/          slice.py, mcp.py (counted), fake.py, test_pydanticai_slice.py
  conftest.py          `robot` fixture parametrized over fake and ros
  measure.py           writes measurements.json
```

| Slice item | LangChain 1.4.9 | PydanticAI 2.67 (`pydantic-ai-slim`) |
|---|---|---|
| 1. READ + ACTUATE tools | `StructuredTool.from_function(coroutine=...)`, `metadata={"effect", "targets"}` | `Tool(fn, metadata={"effect", "targets"})` in a `FunctionToolset` |
| 2. ReAct loop | `create_agent` | `Agent` |
| 3. StateBased context | `AgentMiddleware.awrap_model_call`: `request.override(messages=[*msgs, obs])` | capability `before_model_request`: `replace(request_context, messages=[*msgs, obs])` |
| 4. Approval gate on ACTUATE | `HumanInTheLoopMiddleware(interrupt_on={actuate tools})`, needs a checkpointer, resumed with `Command(resume=...)` | `ApprovalRequiredToolset(func=effect is ACTUATE)`, returns `DeferredToolRequests`, resumed with `DeferredToolResults` and the message history |
| 5. Fail-closed wrapper | `AgentMiddleware.awrap_tool_call` returning `ToolMessage(status="error")`; a second, tool-level `guarded()` for paths without middleware | `WrapperToolset.call_tool` raising `ToolFailed`; the same wrapper covers every path |
| 6. MCP client | `langchain-mcp-adapters` `MultiServerMCPClient` | built in: `MCPToolset` (on fastmcp) |
| 6. MCP server (stdio) | `langchain_mcp_adapters.to_fastmcp` + `mcp` `FastMCP` | not provided: a 40-line toolset-to-MCP adapter on the `mcp` low-level server |
| 7. Cancellation | `task.cancel()` on `agent.ainvoke` | `task.cancel()` on `agent.run` (a `CancellationToken` also exists) |

Tests, the same 25 in both slices (each runs on `fake` and `ros` unless noted). LangChain has 6 more, for the tool-level guard that §5 recommends:
- the tools read `/d1/status` and drive `/cmd_vel`, ending with a zero Twist;
- the ReAct loop runs a tool and answers;
- the StateBased observation is the last message of every model call, appears exactly once per call, and is never in the stored history;
- approve and reject both work, and nothing is published before approval;
- a hook that crashes, hangs (0.2 s timeout) or denies leads to a denied tool result. The run continues, the next READ call still works, and the run finishes in under 2 s;
- cancelling during a 10 s drive returns in under 0.3 s, the tool's cleanup sends a zero Twist, and publishing stops;
- an MCP client tool from an external stdio server runs through the same hooks;
- MCP server stdio end to end (fake, and ros with the sim seeing `/cmd_vel`), with `readOnlyHint`/`destructiveHint` set, and a crashing or denying hook giving `isError`;
- the ordering of the approval gate and the hooks (§3.2);
- LangChain only: the tool-level `guarded()` adapter with no middleware. A crashing, hanging or denying hook gives `ToolMessage(status="error")`, and the run continues.

Run (Lyrical; the spike deps are not in the lock, so `uv sync` removes them and `conftest.py` then skips the slices):

```
uv pip install --python .venv/bin/python "pydantic-ai-slim[mcp]==2.67.0" "langchain-mcp-adapters==0.3.2"
source /opt/ros/lyrical/setup.bash && source <overlay>/install/setup.bash && source install/setup.bash && source .venv/bin/activate
cd rai3-plan && ROS_DOMAIN_ID=31 python -m pytest -p no:cacheprovider d1     # 56 passed in 21.8 s
python -m d1.measure
```

The committed `uv.lock` was not changed.

## 2. Measurements

Python 3.14.4 on Lyrical; langchain 1.4.9, langchain-core 1.7.6, langgraph 1.2.20, langchain-mcp-adapters 0.3.2, pydantic-ai-slim 2.67.0, mcp 1.30.0, fastmcp-slim 3.4.11. Raw data is in `measurements.json`. The latency ranges cover 3 runs of `measure.py`.

| Criterion | LangChain | PydanticAI | Kind |
|---|---|---|---|
| Lines of code (non-blank, non-comment, ruff-formatted; excludes tests, fake models and `common/`) | **144** (slice 107 + mcp 37) | 157 (slice 100 + mcp 57) | measured |
| Shared code, the same for both | 241 | 241 | measured |
| Import time: framework only (median of 9 fresh processes) | 0.35-0.37 s (`langchain.agents`) | **0.30-0.33 s** (`pydantic_ai`) | measured |
| Import time: whole slice including MCP | **0.47-0.51 s** | 0.54-0.57 s | measured |
| Per ReAct step, fake model + READ tool, no middleware (median over about 870 steps) | **2.82-2.89 ms** | 3.38-3.84 ms | measured |
| Per step, with context + hook + approval middleware | **3.19-3.33 ms** | 3.53-4.01 ms | measured |
| Step-time growth from step 10 to step 300 (history length) | 2.8-3.1x | 3.1-3.8x | measured |
| Time from cancel to the run task finishing, during a slow tool (median of 20) | **0.22-0.27 ms** | 0.35-0.53 ms | measured |
| Cancellation reaches the running tool coroutine and its `finally` runs | yes | yes | measured |
| Fail-closed on crash, timeout and deny; run continues | yes | yes | measured |
| One wrapper covers the agent, MCP server and direct paths | yes, with the tool-level `guarded()` (tested). The middleware form misses MCP and direct calls | yes: `WrapperToolset` | measured |
| A call the hook denies is still shown to a human approver | yes (HITL runs in `after_model`, before any tool wrapper) | no (the wrapper order is up to RAI) | measured |
| Approval needs a checkpointer and a thread id | yes | no (stateless, history passed back) | measured |
| StateBased context: code needed and persistence | 7 lines, request-only | 6 lines, request-only | measured |
| MCP server support | `to_fastmcp`; it drops tool annotations, so the slice sets them by hand | none; RAI writes an adapter and builds a `RunContext` by hand | measured |
| Packages installed into a clean 3.14 venv | 58 (`langchain` + `langchain-mcp-adapters`) | 51 (`pydantic-ai-slim[mcp]`) | measured |
| Migration: LangChain usage outside rai_core | none needed | rai_bench 16 files / 33 import lines; rai_whoami 9 / 14 (incl. `langchain_community` FAISS and embeddings); rai_perception 3 / 3; rai_semap 1 / 1; rai_s2s 0; examples 8 / 11 | measured (grep), effort inferred |
| Migration: rai_core | 30 files / 102 import lines; 29 tool classes rewritten in A9 either way | the same, plus `rai.llm` and the bench mocks on another model layer | measured (grep), effort inferred |
| Migration effort | S: tools move to the RAI tool class (A3/A9) | L (D2) before A3 can start | inferred |
| API stability risk | lower: a 1.x line with a stability promise, and the code already runs on it | higher: 2.x after about a year of 1.x, a very large API surface (capabilities, toolsets, durable exec, realtime), deprecations inside hook contexts, and a stderr banner on the first run | inferred |

How to read the latency numbers: both frameworks add about 3 ms per step, while a real LLM call takes 300-3000 ms, so neither framework's overhead matters for speed. Per step, LangChain was 0.3-1.0 ms faster in every run.

## 3. Findings

### 3.1 Where PydanticAI is better

- **Safety composition.** In PydanticAI the toolset is the single seam: `FailClosed(ApprovalRequiredToolset(CombinedToolset([ros_tools, mcp_tools])))`. The same object serves the agent, the MCP server and direct calls, and RAI decides the order. That order means a denied call never reaches a human, and an approved call is checked again when the run resumes.
- **Approval is stateless.** `DeferredToolRequests` plus `DeferredToolResults` with the message history; no checkpointer and no thread id.
- Importing the framework alone is 0.05 s faster, and it pulls fewer packages.

### 3.2 Where LangChain is better, or the gap is small

- **No migration.** rai_bench, rai_whoami, rai_semap, rai_perception and the examples already use LangChain. rai_whoami's RAG pipeline (`langchain_community` FAISS, embeddings, text splitters) has no PydanticAI equivalent, so D2 would mean rewriting it on raw FAISS. The bench's tool-calling validators read LangChain `AIMessage.tool_calls`.
- **The safety gap closes with a tool-level wrapper.** Tested (`test_tool_level_guard_on_agent_path`): a guard on the tool itself that raises `ToolException` with `handle_tool_error=True` gives `ToolMessage(status="error")`, and the run continues. If the guard raises a plain exception instead, the run ends (checked by hand). So RAI puts the fail-closed adapter in its tool class, not in middleware, and gets "one wrapper, every path" like PydanticAI. What remains is the ordering: `HumanInTheLoopMiddleware` asks before any tool-level check. That is a UX cost, not a safety hole: the hook still denies after approval, and the connector-side egress chain (R6) is the real guarantee. `ConfirmActuation` (A8) can be an awaiting hook inside the same wrapper instead of HITL, which removes the ordering problem and also works over MCP; 03 §2.3 already proposed that.
- **MCP server.** `to_fastmcp` exists. PydanticAI has nothing, and the adapter has to build a `RunContext` by hand with a `TestModel`, which is not documented as supported. LangChain's `to_fastmcp` drops annotations, so for A10 RAI registers tools with FastMCP itself, in about 20 lines.
- Per-step latency, the whole-slice import time and cancel latency are all slightly lower. LOC is 13 lower.

### 3.3 Equal

- StateBased context per call, never persisted: 7 and 6 lines, both request-only by design (`request.override` / `replace(request_context, ...)`). Neither framework has today's `/rosout: /rosout:` bug.
- Cancellation: in both, `Task.cancel()` reaches the running tool coroutine within one event-loop tick, so tools only need a `try/finally` to stop the hardware. A synchronous tool run through `to_thread` still cannot be interrupted in either framework, which is why A9 makes action-based tools async.
- ReAct loop size: both are one call.

## 4. Decision

**LangChain 1.x** (`langchain.agents.create_agent` plus middleware, `langchain-mcp-adapters` for the MCP client).

1. Both frameworks implement the whole slice and pass the same 25 tests on Lyrical, on both the fake robot and real ROS, so neither is ruled out on capability.
2. PydanticAI's one real advantage, a single toolset wrapper covering every path, LangChain matches by putting the fail-closed adapter on the RAI tool class (tested, §3.2). Only the HITL-before-hook ordering is left, and an awaiting `ConfirmActuation` hook removes it.
3. LangChain is ahead or level on every measured number except framework-only import time (0.05 s) and package count (58 vs 51): LOC (144 vs 157), whole-slice import time, step latency (0.3-1.0 ms less), cancel latency, and MCP server support.
4. PydanticAI would need D2, a large rewrite (rai_bench, rai_whoami's RAG stack, perception, semap, examples), before A3 can start. It would also add API-churn risk from a 2.x line with a very large surface.
5. The criteria said ties go to LangChain. This isn't even a tie.

## 5. What the design looks like on LangChain

**A3: RAI tool class and fail-closed adapter.**
- `RAITool(BaseTool)`, or a `@rai_tool(effect=..., targets=...)` decorator that builds a `StructuredTool`, stores `effect` and `targets` in `metadata`, and is async-native (`coroutine=`).
- The fail-closed adapter lives inside the tool: `_arun` first runs `rai.hooks.check(hooks, ToolCall(name, args, effect, targets), timeout)`. A denial, a hook crash or a hook timeout raises `ToolException`, and `handle_tool_error=True` turns it into `ToolMessage(status="error")`, so the run continues. Because it is in the tool, it covers the agent, `rai mcp serve`, direct calls and rai_bench.
- Hooks are found through a contextvar or the agent's `HookRegistry`, not passed per tool.
- `from_langchain(tool, effect=...)` wraps third-party tools the same way. MCP client tools get their effect from annotations: `readOnlyHint` gives READ, `destructiveHint` gives ACTUATE, anything else gives WRITE.
- Middleware is used only for agent-only concerns.

**A6: ReAct and StateBased.**
- `ReActAgent` is `create_agent(model, tools, system_prompt=..., middleware=[...], checkpointer=InMemorySaver())`.
- `StateBasedAgent` adds a `StateContext(AgentMiddleware)` whose `awrap_model_call` gathers the aggregator observations (concurrently, each with its own timeout, keeping the last good result). It appends them as the last message via `request.override(messages=...)`, so they are never written to state.
- Approval: `ConfirmActuation` is an opt-in awaiting hook in the tool wrapper. `HumanInTheLoopMiddleware` stays available for users who want LangGraph interrupts.
- Interrupts are `Task.cancel()` on the `ainvoke`/`astream` task, and that reaches the running tool.
- `recursion_limit` replaces `max_steps`.

**A10: MCP.**
- Client: `langchain-mcp-adapters` `MultiServerMCPClient(...).get_tools()`, wrapped with `from_langchain` so the RAI hooks apply.
- Server: `rai mcp serve` registers RAI tools on `mcp.server.fastmcp.FastMCP` directly, in about 20 lines rather than `to_fastmcp`, which drops annotations and turns a handled tool error into a success. It sets `readOnlyHint`/`destructiveHint` from `effect` and maps the error status to `isError`.
- The connector egress chain (R6) still applies under every path.
- Both go in the `rai-core[mcp]` extra (`mcp`, `langchain-mcp-adapters`). The stdio e2e test pattern is `d1/common/mcp_e2e.py`.

## 6. Limits of this spike

- The fake chat models skip provider serialization, so step latency measures framework overhead only.
- Memory use, streaming and multimodal content blocks were not compared.
- Jazzy was not run, because D1 doesn't require it. Both frameworks are pure Python and Jazzy runs Python 3.12, so nothing distro-specific is expected (inferred).
- The PydanticAI MCP server adapter builds a `RunContext` itself. It works on 2.67, but that is not a documented API.
