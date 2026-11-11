# RAI 3.0 plan: agent, tool and LLM layer

Scope: `rai.agents`, `rai.tools`, `rai.aggregators`, `rai.initialization`, `rai.frontend`, `rai.messages`, `rai.types`, `config.toml`, and how `rai_whoami`, `rai_s2s`, `rai_semap`, `rai_perception`, `rai_bench` and `examples/` use them. All paths are relative to `src/rai_core/rai/` unless they start with `src/`, `examples/` or `tests/`. Claims marked **(verified)** were reproduced in the worktree `.venv`.

## 0. Summary of decisions

1. **The tool interface belongs to RAI, and hooks run in one place.** A tool is a `ToolSpec` (name, description, Pydantic input model, `effect`, timeout) plus an implementation that can be sync or async. A single `ToolExecutor` validates the call, runs the hooks, enforces the timeout and normalizes errors. LangChain, MCP and direct calls all go through the executor. LangChain only appears in the `to_langchain()` adapter.
2. **Each tool declares its effect (`READ` / `WRITE` / `ACTUATE`) and the ROS targets it touches.** Nothing is deny-by-default. The labels let a single hook guard every action that moves hardware, and the connector side uses the same `Target` type.
3. **The agent lifecycle is an explicit, async-first state machine.** States are CREATED, STARTING, RUNNING, STOPPING, STOPPED and FAILED. Inside RUNNING, an LLM turn is a separate IDLE/BUSY sub-state. An interrupt is a real `Task.cancel()`, so it also reaches an LLM stream or a tool that is still running. Sync wrappers run on one background event loop.
4. **ReAct and StateBased share one graph.** The graph is a small RAI-owned LangGraph `StateGraph` with the `add_messages` reducer and an `InMemorySaver` checkpointer. StateBased is ReAct plus a `ContextProvider` that adds a fresh observation for each LLM call and does not store it in the history.
5. **Hooks are gate events (a sequential pipeline that can modify, `Deny` or `Halt`) and observe events (concurrent, never block).** Each hook has a priority, a timeout and a fail-closed default. The `HookRegistry` and `Decision` types are shared with the connector side, and the two sides are linked through the `current_call` contextvar.
6. **Aggregators v2 run on demand and are bounded.** Each one has a window and a TTL cache, and computes the observation only when the LLM needs context. VLM aggregators therefore stop costing money while the agent is idle.
7. **Settings use `pydantic-settings`**, with the `RAI_` env prefix and an optional `rai.toml`. Models are referenced as `"provider:model"` and resolved with `langchain.chat_models.init_chat_model`. That removes all vendor branching and the eager provider imports.
8. **Vendors are optional extras.** Tier 1: `openai` (which also covers OpenAI-compatible servers), `anthropic`, `ollama`, `google`. Tier 2: `bedrock`, via Converse. `langchain-community`, `langgraph-prebuilt`, `streamlit`, `tomli` and `tomli-w` leave the core dependencies.
9. **MCP support uses the official `mcp` SDK** (`rai-core[mcp]`). The server exposes the executor, so hooks apply to external MCP clients too. The client wraps remote tools as RAI `Tool`s, so hooks apply to those as well.

---

## 1. Audit

### 1.1 Abstractions and LangChain leakage

| Area | Current design | Problem |
|---|---|---|
| `BaseAgent` (`agents/base.py:19-34`) | Abstract class with sync `run()` and `stop()` and no states | It serves as the base for LLM agents and also for ASR, TTS, vision services and whoami (`src/rai_s2s/rai_s2s/*/agents/*.py`, `src/rai_extensions/rai_perception/rai_perception/agents/*.py`, `src/rai_whoami/rai_whoami/agents/ros2/*.py`). Every subclass re-implements the same thread + `Event` loop (e.g. `examples/agriculture-demo.py:57-77`, where `stop()` never joins). |
| `LangChainAgent` (`agents/langchain/agent.py:45`) | Wraps a `Runnable` in threads, a deque and 3 events | The lifecycle and the "direct mode" are bundled together, so users build an agent only to pull the graph back out: `ReActAgent(target_connectors={}, ...).agent` (`examples/rosbot-xl-demo.py:75-80`). |
| Tools | Every tool subclasses LangChain `BaseTool`. Toolkits subclass `BaseToolkit` (`tools/ros2/base.py:23,120`) | The tool API is LangChain. Config fields (connector, topic) and LLM args live on the same pydantic model. The bench mocks subclass the real tools and override `_run` (`src/rai_bench/rai_bench/tool_calling_agent/mocked_tools.py:57-76`, 18 classes). |
| Messages | `rai.messages` re-exports LangChain messages. `MultimodalMessage` subclasses `BaseMessage` (`messages/multimodal.py:22`) | Images are base64 PNG strings. `ToolMultimodalMessage.postprocess` splits tool and human messages for OpenAI (`multimodal.py:115-135`), and Bedrock has dead code after its `return` (`multimodal.py:137-147`). |
| Aggregators | `get() -> BaseMessage` (`aggregators/base.py:42`) | Sensor summaries are LangChain types. |
| Connectors | `HRIConnector.build_message(AIMessage)` and `HRIMessage.to_langchain()` (`communication/hri_connector.py:20,78,103`) | LangChain inside the communication layer (the connector-side agent owns this). |

33 files in `rai_core/rai` import LangChain. In 3.0, LangChain stays inside `rai.agents`, `rai.llm` (model init) and `rai.tools.langchain` (the adapter). Agent constructors may still accept `BaseChatModel`, because LangChain remains the agent framework.

### 1.2 Implicit state machines and lifecycle bugs

- **`stop()` before `run()` hangs forever (verified).** `stop()` waits on `_agent_ready_event` with no timeout (`agent.py:221`). Only `run()` (`agent.py:167`) or `_run_agent` sets that event. Reproduced: after `LangChainAgent(...).stop()`, the thread is still blocked 3 s later.
- **The state lives in 4 variables**: `_thread is None`, `_stop_event`, `_interrupt_event` and `_agent_ready_event` (`agent.py:137-141`). `ready()` reads the deque without the lock (`agent.py:170-171`).
- **An interrupt only takes effect between LangGraph chunks** (`agent.py:201-208`). An LLM call or a tool call that is already running continues. A blocking navigation goal (`tools/ros2/navigation/nav2_blocking.py:129`) keeps the robot moving after the user says "stop".
- **Polling loops**: `wait()` sleeps 0.1 s per check (`agent.py:174-175`), the idle path sleeps 0.1 s (`agent.py:193`), and `_run_loop` spins on `wait(0.01)` (`agent.py:213-215`). Up to 100 ms of extra latency per message, and some idle CPU use.
- **The conversation history depends on in-place mutation.** `llm_node` returns `None` and mutates `state["messages"]` (`core/react_agent.py:75-85`). `LangChainAgent` never reads the stream output. It relies on LangGraph passing its own list object through (`agent.py:200-206`). That breaks with reducers or checkpointers, and the history grows without bound.
- **`HRICallbackHandler` keeps state across runs** (`agents/langchain/callback.py:49-56`). `hit_on_llm_new_token` is never reset, so the "model did not stream" fallback only fires once per process. The `chunks_buffer` is shared by concurrent LLM runs. When the last token is a split char, the buffer is empty at `on_llm_end`, so no `done=True` chunk is sent (`callback.py:136-139`) and the HRI consumers never see the end of a message.
- **`BaseStateBasedAgent`**: `stop()` duplicates the parent (`state_based_agent.py:185-201` vs `agent.py:217-227`). It also unregisters the aggregator callbacks and shuts down a connector it created itself (`setup_connector`, `agents/ros2/state_based_agent.py:20-21`; this is a second ROS node next to the one the tools use). A `run()` after `stop()` therefore runs with no aggregators.

### 1.3 StateBased context bugs (verified)

`retriever_wrapper` (`core/state_based_agent.py:92-103`) runs before every LLM step:
- It **mutates the cached aggregator message**: `message.content = f"{source}: {message.content}"`. After 3 steps a log state reads `'/rosout: /rosout: /rosout: logs: ok'`.
- It **appends the same object to the permanent history on every step**, so the context grows linearly with the number of steps and turns.
- For image aggregators the content is a list, so the f-string turns it into a string. `pretty_repr()` is evaluated eagerly in the debug log f-string and then **raises `TypeError`**. StateBasedAgent with `ROS2GetLastImageAggregator` crashes on its first step.

### 1.4 Slow paths

| # | Evidence | Cost |
|---|---|---|
| 1 | `import rai` takes **0.7 s (verified)**: `rai/__init__.py` imports `agents`, and `initialization/model_initialization.py:22-28` eagerly imports `langchain_aws`, `langchain_ollama`, `langchain_openai` and `langsmith`. `coloredlogs.install()` runs at import (`:32`), a global side effect | Startup time of every CLI, node and test |
| 2 | `invoke_llm_with_tracing` calls `get_tracing_callbacks()` on **every LLM call** (`agents/langchain/invocation_helpers.py:54`). That reads `config.toml` from the CWD and builds a `Langfuse()` client (`model_initialization.py:363,380-387`). `get_llm_model` also re-reads the file on every call (`:168`) | File I/O plus client construction per step. The handlers are probably duplicated, since the config callbacks are already inherited (`invocation_helpers.py:66-68`) |
| 3 | Aggregators convert **every** buffered image to base64 but use 1 or 3 of them (`aggregators/ros2/aggregators.py:90,144`). The buffer is unbounded by default (`aggregators/base.py:29`) | At 30 Hz for 5 s, 150 PNG encodes per tick |
| 4 | VLM aggregators run every `time_interval` forever, even when no one asks (`state_based_agent.py:133-147`) | Continuous VLM spend while the agent is idle |
| 5 | `store_artifacts` reads and rewrites a whole pickle DB in the CWD for every image tool call (`messages/artifacts.py:25-42`) | O(n) per call, unbounded file, unsafe pickle |
| 6 | `ToolRunner` forces `max_concurrency=1` (`core/tool_runner.py:61-63`) | Read-only tools are serialized too |
| 7 | A new ROS client per call that is never destroyed (`tools/ros2/manipulation/custom.py:88,220,424`; `nav2_blocking.py:112`), plus `wait_for_service` every time | DDS entity leak, and discovery latency on each call |

### 1.5 Safety gaps

1. **Hardware writes have no guard.** With the default `writable=None`, `PublishROS2MessageTool` publishes to any topic, including `/cmd_vel` (`tools/ros2/generic/topics.py:92-100`). The flagship examples give the LLM the full `ROS2Toolkit` (`examples/agents/react_ros2.py:32`). There is no confirm, rate-limit, e-stop or audit layer anywhere.
2. **The CLI tools bypass the access lists**: `ros2 topic pub`, `ros2 service call`, `ros2 param set/delete`, `ros2 action send_goal` (`tools/ros2/cli.py:73-189`). The only check is a character blacklist (`cli.py:21,44`).
3. **The timeout decorator does not time out (verified).** `with ThreadPoolExecutor()` waits for the worker on `__exit__` (`tools/timeout.py:87-96`). `@timeout(0.2)` around `sleep(2)` returned after 2.0 s. `rai_perception`'s gripping-point tool depends on it (`src/rai_extensions/rai_perception/rai_perception/tools/gripping_points_tools.py:193-260`). The test only checks that the exception is raised, not the timing (`tests/tools/test_timeout.py:52-57`).
4. **Nav2 tool state is held in module globals** (`tools/ros2/navigation/nav2.py:34-36,136-137`). A second goal overwrites `current_action_id`, so the first goal can no longer be cancelled. The state is also shared across agents and robots. The generic action stores are module globals as well (`tools/ros2/generic/actions.py:29-33`).
5. **Stopping or interrupting does not stop the motion** (see 1.2). In `MoveObjectFromToTool`, if the second move fails, the arm is not reset (`custom.py:280-316`).

### 1.6 Duplicated and dead code

- `ReActAgentState` and `llm_node` are copied word for word (`core/react_agent.py:36-85` and `core/state_based_agent.py:40-89`), and there is a third variant in `core/conversational_agent.py:40-65`.
- The vendor `if/elif` chain appears 3 times (`model_initialization.py:179-217, 220-259, 262-356`).
- `wait_for_shutdown` exists twice (`agents/runner.py:26-54` and `:122-141`). `run_agents` and `AgentRunner.run` are the same.
- The connector, readable, writable and forbidden plumbing is repeated in every toolkit (`topics.py:38-76`, `services.py:29-43`, `actions.py:73-103`, `toolkit.py:23-50`).
- The manipulation tools copy client setup, wait and pose building 3 times (`custom.py`).
- `timeout` and `timeout_method` are near-duplicates (`tools/timeout.py:49-161`).
- Dead or misplaced code: `core/plan_agent.py` (exported nowhere and used nowhere). `create_conversational_agent` is deprecated "for removal in 3.0" (`core/conversational_agent.py:68-71`) but still used by `rai_bench` and 3 examples. `megamind` is used only by a `rai_bench` example. `structured_output_agent` is used only by the `vlm_benchmark`.

### 1.7 Dependencies (`src/rai_core/pyproject.toml`)

- `langchain-community` is not imported anywhere in `rai_core`. Only `rai_whoami` uses it, and that package already declares it. Drop it from core.
- `langgraph-prebuilt` is only used for `tools_condition` and `msg_content_output` (trivial) and for `create_react_agent` in megamind (deprecated in LangGraph 1.x). Drop it.
- `langchain` is only used for `from langchain.tools import BaseTool` (`topics.py:22`). Keep it for `init_chat_model` and `init_embeddings` (verified importable with `langchain` 1.4.9).
- `streamlit` (heavy) is a core dependency. Move it to an extra.
- `tomli` and `tomli-w`: use stdlib `tomllib`. `tomli-w` only serves the configurator, which is dropped.
- `deprecated`: use `typing_extensions.deprecated`, because `warnings.deprecated` is 3.13+ and Jazzy runs 3.12.
- Locked versions: `langchain-core` 1.7.6, `langgraph` 1.2.20, `pydantic-settings` 2.16.2 (already transitive).

---

## 2. 3.0 design

### 2.1 Package layout (rai_core)

```
rai/
  settings.py          # RAISettings, get_settings(), section()
  llm.py               # get_model(), get_embeddings(), tracing_handler()
  runtime.py           # background loop, run(*agents), signal handling
  hooks/               # core.py (shared with connectors), events.py, builtin/ (estop, confirm, ratelimit, audit)
  tools/               # spec.py, executor.py, decorators.py, langchain.py (adapter), mcp.py, ros2/...
  agents/              # base.py (state machine), llm_agent.py, graph.py, react.py, state_based.py, outputs.py
  aggregators/         # base.py, context.py, ros2.py
  messages.py          # content-block helpers, Image type; no LangChain subclasses
```

`rai/__init__.py` exports nothing heavy. It uses lazy `__getattr__`, so `import rai` does not import LangChain.

### 2.2 Tool interface

```python
# rai/tools/spec.py  (no LangChain import)
class Effect(StrEnum):
    READ = "read"         # observes only
    WRITE = "write"       # changes software state (params, services, maps)
    ACTUATE = "actuate"   # can move hardware

@dataclass(frozen=True)
class Target:             # shared with the connector side
    kind: Literal["topic", "service", "action", "param", "other"]
    name: str
    msg_type: str | None = None

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    effect: Effect = Effect.READ
    timeout_s: float | None = 30.0                     # None = no limit (use with care)
    tags: frozenset[str] = frozenset()
    # Optional per-call classification for generic tools (publish to <topic>, ros2 cli ...)
    classify: Callable[[BaseModel], tuple[Effect, tuple[Target, ...]]] | None = None

@dataclass
class ToolResult:
    text: str = ""
    images: tuple[Image, ...] = ()       # rai.messages.Image(bytes|b64, mime)
    data: Any = None                     # JSON-able, becomes MCP structuredContent
    is_error: bool = False

class ToolContext:                       # injected if the impl asks for it
    agent: str; call_id: str; logger: logging.Logger
    async def progress(self, msg: str) -> None: ...   # streamed as AgentEvent / MCP progress

class Tool(Protocol):
    spec: ToolSpec
    async def run(self, ctx: ToolContext, args: BaseModel) -> ToolResult: ...
```

There are two ways to write a tool. Both produce a `Tool`:

```python
@tool(effect=Effect.ACTUATE, timeout_s=120, tags={"nav"})
async def navigate_to_pose(ctx: ToolContext, x: float, y: float, yaw: float = 0.0) -> str:
    """Drive the base to (x, y, yaw) in the map frame. Blocks until the goal ends."""
    ...                     # input model is derived from the signature + docstring

class GetCameraImage(ToolBase):           # configured tool: config in __init__, LLM args in Input
    class Input(BaseModel):
        pass
    spec = ToolSpec("get_camera_image", "Get the latest camera image", Input)
    def __init__(self, connector: ROS2Connector, topic: str): ...
    async def run(self, ctx, args) -> ToolResult: ...
```

Rules:
- The return value is coerced: `str` becomes text, `BaseModel`/`dict` becomes JSON text plus `data`, `ToolResult` passes through.
- A sync implementation runs through `asyncio.to_thread`, which copies the contextvars. On timeout, that thread cannot be killed, so long or hardware-moving tools must be async.
- On `CancelledError`, an async tool must clean up, for example cancel the action goal. A helper, `ros2.ActionCall`, does this for every action-based tool.
- Splitting the spec from the implementation makes mocks trivial: `spec.with_impl(fake_fn)`. The 18 bench mock classes become functions.

**Executor.** All hooks, timeouts and errors are handled here:

```python
class ToolExecutor:
    def __init__(self, tools: Sequence[Tool], hooks: HookRegistry, *, agent: str): ...
    async def call(self, name: str, raw_args: dict, *, call_id: str) -> ToolResult:
        # 1 lookup  -> error ToolResult if unknown
        # 2 validate raw_args with input_model -> compact ValidationError text (moved from tool_runner.py:97-117)
        # 3 effect, targets = spec.classify(args) if classify else (spec.effect, ())
        # 4 call = ToolCall(...); token = current_call.set(call)
        # 5 decision = await hooks.gate("tool.before", call)   -> Deny => error ToolResult, Halt => raise AgentHalted
        # 6 async with asyncio.timeout(spec.timeout_s): result = await tool.run(ctx, args)  (ACTUATE: under self._actuate_lock)
        # 7 result = await hooks.gate("tool.after", outcome)    (may redact/modify)
        # 8 hooks.emit("tool.end", outcome)  -- always, incl. denied/timeout/error, with decided_by
    async def call_batch(self, calls) -> list[ToolResult]:
        # READ/WRITE calls run concurrently; ACTUATE run sequentially in LLM order
    def cancel_actuation(self) -> None:   # used by e-stop: cancels in-flight ACTUATE tasks
```

**LangChain adapter** (`rai/tools/langchain.py`, about 60 lines):

```python
def to_langchain(tool: Tool, executor: ToolExecutor) -> StructuredTool:
    async def _call(**kw):
        r = await executor.call(tool.spec.name, kw, call_id=<from injected ToolCallId>)
        return r.text, {"images": r.images, "data": r.data}
    return StructuredTool(name=..., description=..., args_schema=tool.spec.input_model,
                          coroutine=_call, response_format="content_and_artifact")

def from_langchain(t: BaseTool, *, effect: Effect = Effect.READ, timeout_s=30.0) -> Tool: ...
```

The RAI graph does not need `to_langchain` to execute tools; it calls the executor directly. `to_langchain` exists for users who bring their own LangGraph or `create_agent`, and for `rai_bench`. Turning images into provider messages (the OpenAI "tool + human message" split) happens only in the graph's tool node. `ToolMultimodalMessage` and the pickle artifact store are removed.

**MCP adapter** (`rai/tools/mcp.py`, `rai-core[mcp]`, official `mcp` SDK):

```python
def mcp_server(executor: ToolExecutor, *, name="rai") -> mcp.server.lowlevel.Server:
    # list_tools: ToolSpec -> mcp.types.Tool(inputSchema=input_model.model_json_schema(),
    #             annotations=ToolAnnotations(readOnlyHint=effect==READ, destructiveHint=effect==ACTUATE))
    # call_tool:  executor.call(...)  -> TextContent + ImageContent (+ structuredContent=data, isError)
async def serve(tools, *, hooks, transport: Literal["stdio", "streamable-http"] = "stdio", **kw): ...

async def load_mcp_tools(session: mcp.ClientSession, *, effect_default=Effect.WRITE) -> list[Tool]:
    # each remote tool becomes a RAI Tool; effect from annotations (readOnlyHint -> READ,
    # destructiveHint -> ACTUATE, else WRITE per MCP default semantics); input model built from JSON schema
```

Because the server is a thin layer over the same executor, any hook installed on that executor applies to external MCP clients such as Claude Desktop or IDEs. Rejected: depending on `langchain-mcp-adapters`. Its tools are LangChain tools, so RAI hooks would not run on them.

### 2.3 Hooks API

**Shared core** (`rai/hooks/core.py`, also used by the connector-side design):

```python
@dataclass(frozen=True) class Allow: ...
@dataclass(frozen=True) class Modify(Generic[P]): value: P        # replace payload (args/messages/result)
@dataclass(frozen=True) class Deny: reason: str                   # this action is refused; LLM sees reason
@dataclass(frozen=True) class Halt: reason: str                   # refuse + abort the turn (e-stop semantics)
Decision = Allow | Modify | Deny | Halt
HookFn = Callable[[P], Decision | None | Awaitable[Decision | None]]

class Priority(IntEnum):        # convention, any int allowed
    SAFETY = -100               # e-stop, bounds  (runs first)
    POLICY = 0                  # rate limits, access
    CONFIRM = 100               # human confirmation last: never ask for something that will be denied

class HookRegistry:
    def on(self, event: str, fn: HookFn | None = None, *, priority: int = 0,
           when: Callable[[P], bool] | None = None, timeout_s: float = 2.0,
           on_failure: Literal["deny", "allow"] = "deny") -> Unsubscribe | decorator: ...
    async def gate(self, event: str, payload: P) -> GateResult[P]:
        # sequential, ascending priority, ties by registration order;
        # each hook sees the payload as modified by previous hooks; Deny/Halt short-circuits.
        # timeout or exception => on_failure (fail-closed by default), logged with hook name.
    def emit(self, event: str, payload: P) -> None:
        # observe-only: hooks run concurrently as tasks; errors logged; never blocks the caller
    def gate_blocking(self, event, payload, timeout_s) -> GateResult[P]:
        # for sync connector code paths: runs gate() on rai.runtime loop
```

**Agent-side events** (lean set):

| Event | Kind | Payload | Typical use |
|---|---|---|---|
| `agent.state` | observe | `StateChange(agent, old, new)` | audit, health |
| `input` | gate | `InputEvent(agent, HRIMessage)` | drop or redact input, "stop!" keyword triggers e-stop (`Halt`) |
| `llm.before` | gate | `LLMCall(agent, model, messages, turn, step)` | rate or budget limit (may `await asyncio.sleep`), trim, inject |
| `llm.after` | gate | `LLMResponse(call, AIMessage, usage)` | strip forbidden tool calls, token accounting |
| `tool.before` | gate | `ToolCall(agent, call_id, spec, args, effect, targets, approvals)` | confirm, veto, clamp args (`Modify`), rate limit, e-stop |
| `tool.after` | gate | `ToolOutcome(call, result, duration)` | redact or post-process the result |
| `tool.end` | observe | `ToolOutcome(..., status, decided_by)` | audit log (always fires: ok, denied, timeout, error) |

After-gates run in descending priority, like middleware layers, so a SAFETY hook sees the final result.

**How `Deny` and `Halt` behave in the agent.** `Deny` turns into an error `ToolResult`: `"Denied by <hook>: <reason>"`. The LLM can then re-plan. `Halt` raises `AgentHalted` inside the turn. The turn task is cancelled, `executor.cancel_actuation()` runs, the agent emits `TurnEnd(status="halted")`, and the agent returns to IDLE. The agent is not stopped.

**Composing with connector-side hooks.**
- `rai.hooks.context.current_call: ContextVar[ToolCall | None]` is set by the executor around `tool.run`. `asyncio.to_thread` copies the context, so a connector `send_message` running inside a tool can see which agent, tool and call caused the write. The connector side uses it for attribution (audit), to skip a second confirmation when `"confirm" in call.approvals`, and optionally to treat writes outside any tool call differently.
- `Target` and `Decision` are the same types on both sides. One policy object installs on both registries:

```python
estop = EStop()                                   # rai.hooks.builtin
estop.install(agent.hooks, connector.hooks)       # agent: Deny ACTUATE + Halt running turn
                                                  # connector: Deny sends to actuation targets
estop.trigger_on_topic(connector, "/rai/estop")   # or estop.trigger("operator") from code / HRI keyword
```

**Built-in hooks** (`rai.hooks.builtin`, each about 50-100 lines):
- `EStop`: a latch. It denies `ACTUATE` calls and actuation targets, halts the current turn, cancels in-flight actuation, and is reset explicitly.
- `ConfirmActuation(ask, match=ACTUATION_DEFAULT, timeout_s=30)`: `ask` is `async (ToolCall) -> bool`. Implementations are provided for an HRI connector (asks the human, waits for "yes"/"no") and for a stdin CLI. Timing out means deny. `ACTUATION_DEFAULT` matches `effect == ACTUATE` or targets like `*/cmd_vel*`, `*joint_trajectory*`, `navigate_to_pose`, `*/follow_joint_trajectory`, so the generic publish and CLI tools are covered as well.
- `RateLimit(max_calls, per_s, match=...)`: a token bucket that returns `Deny("rate limited, retry in 1.2s")`. Can also run on `llm.before` in delay mode.
- `AuditLog(sink)`: JSONL through a bounded queue. It records tool calls, decisions, durations and LLM usage, plus images if requested (this replaces the pickle store).

Rejected:
- A permission DSL, deny-by-default, and LLM-judge safety hooks (a user can write those as a hook).
- LangGraph `interrupt()` for confirmation. It needs a checkpointer and resume plumbing, and it would not work on the MCP-server path. An awaiting hook is simpler and works everywhere.
- LangChain 1.x `AgentMiddleware` as the hook mechanism. It is not reachable from MCP or direct calls.

### 2.4 Agent base and lifecycle state machine

```python
class AgentState(StrEnum):
    CREATED = "created"; STARTING = "starting"; RUNNING = "running"
    STOPPING = "stopping"; STOPPED = "stopped"; FAILED = "failed"

_ALLOWED = {
    CREATED:  {STARTING, STOPPED},          # stop() before start() is a no-op -> STOPPED (fixes hang)
    STARTING: {RUNNING, FAILED, STOPPING},
    RUNNING:  {STOPPING, FAILED},
    FAILED:   {STOPPING},
    STOPPING: {STOPPED},
    STOPPED:  set(),                        # terminal; no restart (create a new agent)
}

class BaseAgent(ABC):
    def __init__(self, *, name: str | None = None, hooks: HookRegistry | None = None): ...
    state: AgentState                                   # read-only property
    async def start(self) -> None:                      # CREATED->STARTING->on_start()->RUNNING; spawns run()
    async def stop(self, timeout_s: float = 10) -> None:# idempotent; cancels run() task, awaits on_stop()
    async def wait(self) -> None:                       # until STOPPED/FAILED
    async def on_start(self) -> None: ...               # subscribe, create services
    async def on_stop(self) -> None: ...                # unsubscribe, release
    async def run(self) -> None: await asyncio.Event().wait()   # optional main body
    async def __aenter__/__aexit__                     # start/stop
    # sync facade (runs on rai.runtime background loop):
    def start_sync(self) / stop_sync(self) / wait_sync(self)

class ThreadedAgent(BaseAgent):                         # migration shim for s2s / perception / whoami
    def loop(self, stop: threading.Event) -> None: ...  # blocking body, run via to_thread
```

Every transition goes through `_transition()`. It checks the table, raises `InvalidTransition`, and emits `agent.state`. If `run()` raises, the agent goes to FAILED and the error is logged.

`rai.runtime`:
- `rai.run(*agents)` is the sync entry point. It calls `asyncio.run`, starts all agents, installs SIGINT/SIGTERM handlers with `loop.add_signal_handler`, and stops the agents in reverse order. It replaces `AgentRunner`, `run_agents` and both copies of `wait_for_shutdown`.
- `runtime.loop()` is a lazily started background loop thread for the sync facades and for `submit_threadsafe`. Connector callbacks run on a `ThreadPoolExecutor` (`communication/base_connector.py:239`), so they hand work to the loop with `loop.call_soon_threadsafe`.

### 2.5 LLM agents: ReAct and StateBased

```python
class InputPolicy(BaseModel):            # replaces the 5-value newMessageBehaviorType (agent.py:36-42)
    on_busy: Literal["queue", "interrupt"] = "interrupt"
    take: Literal["all", "last", "first"] = "last"
    max_pending: int = 100

AgentEvent = TextDelta | ToolStarted | ToolFinished | TurnEnd   # TurnEnd.status: done|interrupted|halted|error

class LLMAgent(BaseAgent):
    def __init__(self, graph: CompiledStateGraph, *, hooks=None, policy=InputPolicy(),
                 outputs: Sequence[OutputSink] = (), thread_id: str = "default"): ...
    def listen(self, connector: HRIConnector, source: str) -> None   # replaces subscribe_source
    def submit(self, msg: HRIMessage | str) -> None                  # thread-safe, non-blocking
    async def interrupt(self) -> None                                # cancels the current turn task
    async def astream(self, msg, *, thread_id=None) -> AsyncIterator[AgentEvent]   # direct mode
    async def ainvoke(self, msg, *, thread_id=None) -> TurnResult                  # direct mode
    def invoke(self, msg, **kw) -> TurnResult                                      # sync wrapper
    async def run(self):                       # main loop: no polling
        while True:
            batch = await self._inbox.take(self.policy)          # asyncio.Condition-based
            self._turn = asyncio.create_task(self._run_turn(batch))
            await asyncio.wait([self._turn])                     # interrupt => turn cancelled, loop continues
```

The direct mode does not need `start()`: `ainvoke`/`astream` run the turn in the caller's loop. This removes the `ReActAgent(target_connectors={}).agent` workaround.

`_run_turn` wraps `graph.astream(..., stream_mode=["messages", "updates", "custom"])`. It converts the stream to `AgentEvent`s and fans them out to `outputs`. Tracing callbacks (Langfuse) are attached once per turn in `RunnableConfig`. LangChain propagates the config through contextvars on Python 3.11+, so `invoke_llm_with_tracing` is removed.

Outputs:
- `HRIOutput(connector, target, chunker=SentenceChunker(max_chars=200))` keeps chunk state per turn and always sends a final chunk with `seq_end=True`. It replaces `HRICallbackHandler`.
- `StreamlitOutput(container)` replaces `StreamlitCallbackHandler`.

**One graph builder** (`rai/agents/graph.py`, about 120 lines):

```python
class ContextProvider(Protocol):
    async def observe(self) -> list[BaseMessage]: ...   # ephemeral, NOT stored in history

def build_react_graph(model: BaseChatModel, executor: ToolExecutor, *, system_prompt: str | SystemMessage = "",
                      context: Sequence[ContextProvider] = (), hooks: HookRegistry,
                      max_history_tokens: int | None = None, checkpointer=None) -> CompiledStateGraph:
    bound = model.bind_tools([s.to_json_schema() for s in executor.specs])
    async def llm(state: MessagesState, config):
        history = trim(state["messages"], max_history_tokens)
        obs = [m for c in context for m in await c.observe()]          # volatile data last: keeps prompt-cache prefix
        call = (await hooks.gate("llm.before", LLMCall(messages=[sys, *history, *obs], ...))).payload
        ai = await bound.ainvoke(call.messages, config)
        ai = (await hooks.gate("llm.after", LLMResponse(call, ai))).payload.message
        return {"messages": [ai]}                                       # reducer, no in-place mutation
    async def act(state: MessagesState):
        results = await executor.call_batch(state["messages"][-1].tool_calls)
        return {"messages": to_provider_tool_messages(results, model)}  # image split only where needed
    # START -> llm -> (tool_calls ? act : END); act -> llm; checkpointer=InMemorySaver() by default
```

```python
class ReActAgent(LLMAgent):
    def __init__(self, model: BaseChatModel | str = "complex", tools: Sequence[Tool | BaseTool] = (), *,
                 system_prompt="", context: Sequence[ContextProvider] = (), max_steps: int = 25,
                 hooks: HookRegistry | None = None, **agent_kw): ...
    graph: CompiledStateGraph   # public: usable by bench / custom LangGraph composition

class StateBasedAgent(ReActAgent):
    def __init__(self, model, tools, *, connector: BaseConnector, aggregators: Sequence[Aggregator], **kw):
        self.observations = AggregatorContext(connector, aggregators)   # ContextProvider
        super().__init__(model, tools, context=[self.observations, *kw.pop("context", ())], **kw)
    async def on_start(self): await self.observations.subscribe()
    async def on_stop(self):  await self.observations.unsubscribe()
```

`ROS2StateBasedAgent` and `setup_connector()` are removed. The connector is passed in, so there is no second node. `max_steps` becomes the LangGraph `recursion_limit`.

What moves out of core:
- `create_megamind` moves to `rai_bench` (its only user).
- `create_structured_output_runnable` moves to `rai_bench/vlm_benchmark` (in practice it is `model.with_structured_output`).
- `create_conversational_agent` is deleted. Bench and examples switch to `ReActAgent(...).graph`.
- `plan_agent.py` is deleted.

### 2.6 Aggregators v2

```python
@dataclass(frozen=True)
class Observation:
    source: str
    text: str = ""
    images: tuple[Image, ...] = ()
    stamp: float = 0.0          # time.monotonic() of newest input; rendered as "observed 2.1 s ago"

class Aggregator(ABC, Generic[T]):
    def __init__(self, source: str, *, msg_type: str | None = None, window: int = 30,
                 max_age_s: float = 5.0, timeout_s: float = 10.0, consume: bool = False): ...
    def push(self, msg: T) -> None            # thread-safe deque(maxlen=window); marks dirty. Cheap.
    @abstractmethod
    async def summarize(self, msgs: Sequence[T]) -> Observation | None: ...
    async def observe(self) -> Observation | None:
        # cached: recompute only if dirty AND cache older than max_age_s; per-call timeout; errors -> stale + warning

class AggregatorContext(ContextProvider):
    async def subscribe(self) / unsubscribe(self)               # connector.register_callback(..., agg.push, raw=True)
    async def observe(self) -> list[BaseMessage]:               # gather(observe()) -> one HumanMessage w/ content blocks
```

- Bounded by default (`window=30`). Images are converted only for the frames that are actually used: the last one for the description aggregator, 3 keyframes for the diff aggregator.
- Work happens on demand, so nothing is computed while idle. The trade-off is that the first LLM step of a turn pays the VLM latency. Later steps within `max_age_s` hit the cache. A background refresh is not included (lean); add it later only if latency matters.
- LLM-based aggregators take `model: BaseChatModel | str = "simple"`, which is resolved lazily. The import-time `get_llm_model` call in `__init__` goes away (`aggregators.py:81,125`).
- Fix the `ROS2LogsAggregator` repeat-collapse: it currently replaces the new line with "repeated N times" (`aggregators.py:46-47`). `consume=True` keeps the "logs since the last look" semantics.
- Observations go through `ContextProvider` and are never written into history. This fixes 1.3.

### 2.7 Settings model

```python
# rai/settings.py
class ModelSettings(BaseModel):
    complex: str = "openai:gpt-4o"                     # "provider:model", init_chat_model syntax
    simple: str = "openai:gpt-4o-mini"
    embeddings: str = "openai:text-embedding-3-small"
    aliases: dict[str, str] = {}                       # extra roles, e.g. {"vlm": "google_genai:gemini-..."}
    provider_kwargs: dict[str, dict[str, Any]] = {}    # {"ollama": {"base_url": ...}, "bedrock_converse": {"region_name": ...}}

class TracingSettings(BaseModel):
    langfuse: bool = False
    langfuse_host: str | None = None                   # keys stay in LANGFUSE_* env vars
    project: str = "rai"
    # LangSmith: no RAI code; native LANGSMITH_TRACING/LANGSMITH_API_KEY env vars work as-is

class ToolSettings(BaseModel):
    default_timeout_s: float = 30.0

class RAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAI_", env_nested_delimiter="__", extra="forbid")
    models: ModelSettings = ModelSettings()
    tracing: TracingSettings = TracingSettings()
    tools: ToolSettings = ToolSettings()
    # sources (high -> low): init kwargs, env, .env, TOML file (RAI_CONFIG_FILE or ./rai.toml if present), defaults

@cache
def get_settings() -> RAISettings: ...                 # get_settings.cache_clear() in tests
def section(name: str, model: type[M]) -> M: ...       # other packages: section("asr", ASRSettings), same file/env

# rai/llm.py
def get_model(ref: str = "complex", **kw) -> BaseChatModel:      # role or "provider:model"
    spec = settings.models.aliases.get(ref) or getattr(settings.models, ref, ref)
    provider = spec.split(":", 1)[0]
    return init_chat_model(spec, **settings.models.provider_kwargs.get(provider, {}), **kw)
def get_embeddings(ref: str = "embeddings", **kw) -> Embeddings: ...  # init_embeddings
```

Example: `RAI_MODELS__COMPLEX=ollama:llama3.1:70b`. Secrets never go in the settings file; each provider's own env var is used (`OPENAI_API_KEY`, ...). `rai_s2s` defines `ASRSettings`/`TTSSettings` read through `section("asr", ...)`, so core no longer knows about audio (today `[asr]`/`[tts]` live in the core template, `initialization/config_initialization.py:61-77`).

The streamlit configurator (`frontend/configurator.py`, 1064 lines) and `rai-config-init` are replaced by a small CLI:
- `rai settings init` writes a commented `rai.toml`.
- `rai settings check` resolves each role and sends a 1-token ping.

### 2.8 Vendor list

| Extra | Package | Tier | Reason |
|---|---|---|---|
| `openai` | `langchain-openai` | 1 | Most used. Through `base_url` it also covers vLLM, LM Studio, OpenRouter and Azure-compatible gateways |
| `anthropic` | `langchain-anthropic` | 1 | New. Strong tool use and vision. Today it is only reachable via Bedrock |
| `ollama` | `langchain-ollama` | 1 | Local and offline robots, edge deployment |
| `google` | `langchain-google-genai` | 1 | Gemini VLMs, already used by defaults |
| `bedrock` | `langchain-aws` (`ChatBedrockConverse`) | 2 | Enterprise AWS users. Use Converse, not legacy `ChatBedrock` (`model_initialization.py:197-205`). Smoke-tested only |
| (none) | any other `init_chat_model` provider | unsupported | Works if the user installs the package; no RAI code |

`langchain-community` is dropped from core. Core installs no provider package. The `rai-core[all-llm]` extra is for demos.

### 2.9 Messages

- Remove the `MultimodalMessage` subclasses. Replace them with helpers that build standard LangChain content blocks: `human(text, images=[...]) -> HumanMessage`.
- `rai.messages.Image` (bytes plus mime) is the RAI-side type used by `ToolResult`, `Observation` and `HRIMessage`. `preprocess_image` stays.
- `to_langchain`/`from_langchain` on `HRIMessage` move to `rai.agents` (coordinate with the connector-side agent).

---

## 3. Breaking changes

1. `config.toml` is gone. Use `RAISettings` (env plus an optional `rai.toml` in a new schema). `get_llm_model("complex_model")` becomes `get_model("complex")`, `get_llm_model_direct(name, vendor)` becomes `get_model("vendor:name")`, and `get_llm_model_config_and_vendor` and `get_tracing_callbacks` are removed. The `rai-config-init` script and the streamlit configurator are removed.
2. Provider packages, `streamlit` and `mcp` become extras.
3. Tools are no longer LangChain `BaseTool`s. `rai.tools.ros2.*` keep their names but take `ROS2Access(readable, writable, forbidden)` instead of 3 fields, and LLM-facing `Input` models are separate from config. Use `to_langchain()` for LangChain consumers. `BaseROS2Toolkit.get_tools()` returns RAI `Tool`s. `rai.tools.timeout` is removed (the executor enforces timeouts). The module-level action and Nav2 stores are removed.
4. `ROS2CLIToolkit` is split: `ROS2CLIReadToolkit` (list/info/echo...) and `ROS2CLIWriteToolkit` (pub/call/set/send_goal, `effect=WRITE`, with `classify` returning targets).
5. `BaseAgent` is async, with an explicit state machine, `start/stop` coroutines, and the `_sync` facades. `run()` now means "main body", not "start". STOPPED is terminal. `AgentRunner`, `run_agents` and `wait_for_shutdown` become `rai.run(...)`. Thread-based subclasses move to `ThreadedAgent`.
6. `LangChainAgent` is removed and becomes `LLMAgent(graph, ...)`. `target_connectors` becomes `outputs=[HRIOutput(...)]`, `subscribe_source` becomes `listen`, `__call__` becomes `submit`, and `new_message_behavior` becomes `InputPolicy`. `HRICallbackHandler` is removed.
7. `ReActAgent`/`StateBasedAgent` constructors change (see 2.5). `BaseStateBasedAgent`, `StateBasedConfig`, `ROS2StateBasedAgent` and `setup_connector` are removed. `create_react_runnable`/`create_state_based_runnable` become `build_react_graph` (or `ReActAgent(...).graph`). The graph state becomes `MessagesState` with a reducer.
8. `create_conversational_agent`, `plan_agent` and `invoke_llm_with_tracing` are deleted. `megamind` and `structured_output_runnable` move to `rai_bench`.
9. Aggregators: `BaseAggregator.get()` becomes `async summarize()`/`observe() -> Observation`. A source is set on the aggregator. Buffers are bounded by default. VLM aggregators are on demand.
10. `rai.messages`: no more LangChain re-exports and no more `*MultimodalMessage`, `ToolMultimodalMessage`, `store_artifacts` or `get_stored_artifacts`.
11. `rai.frontend.run_streamlit_app` and `rai.agents.integrations.streamlit` are reimplemented on `AgentEvent` and live under `rai-core[streamlit]`.
12. `import rai` no longer re-exports `ReActAgent`, `get_llm_model`, etc. eagerly (lazy attribute access instead).

---

## 4. Work items

Sizes are for agentic implementation plus human review: **S** is under 1 day, **M** is 1-3 days, **L** is 3-7 days.

| ID | Item | Size | Depends on | Notes |
|---|---|---|---|---|
| A1 | `rai.settings` + `rai.llm` (`init_chat_model`), extras split, lazy `rai/__init__`, drop coloredlogs-at-import, `rai settings init/check` | M | none | Unblocks every `get_llm_model` caller (whoami 4 files, bench, examples) |
| A2 | Hooks core: `HookRegistry`, `Decision`, gate/emit semantics, `current_call` contextvar | M | none | **Freeze the interface early with the connector-side agent** (shared module) |
| A3 | Tool core: `ToolSpec`, `@tool`, `ToolBase`, `ToolResult`, `ToolExecutor` (validation, timeouts, effect-aware concurrency, cancel) | M | A2 | Replaces `ToolRunner`, `rai.tools.timeout` |
| A4 | LangChain adapter `to_langchain`/`from_langchain` + provider tool-message conversion | S | A3 | |
| A5 | Agent runtime: `BaseAgent` state machine, `ThreadedAgent`, `rai.runtime`, `rai.run` | M | none | Port s2s (3), perception (3), whoami (2), examples (2) to it: +M |
| A6 | `LLMAgent` + `build_react_graph` + `ReActAgent` + `HRIOutput` + `AgentEvent` stream + tracing-once | L | A1, A3, A4, A5 | Core of the release |
| A7 | Aggregators v2 + `AggregatorContext` + `StateBasedAgent` | M | A1, A6 | |
| A8 | Built-in hooks: `EStop`, `ConfirmActuation` (HRI + CLI askers), `RateLimit`, `AuditLog` | M | A2, A3, A6 | `EStop.install` connector half needs the connector registry |
| A9 | Port ROS 2 tools (~35 classes): effects/targets, `ROS2Access`, async `ActionCall` with cancel, no globals, cached clients, CLI split | L | A3, connector async API | The largest single item; mechanical but must be tested in sim |
| A10 | MCP server + client | M | A3, A2 | Extra `rai-core[mcp]`; e2e test with stdio transport |
| A11 | Messages cleanup (content blocks, `Image`, remove pickle store) | S | A4, A6 | Coordinate `HRIMessage` changes |
| A12 | Streamlit integration on `AgentEvent` (extra) | S | A6 | |
| A13 | Removals/moves: plan_agent, conversational, megamind, structured_output -> bench; port `rai_bench` (mocks via `spec.with_impl`), `rai_semap`, `rai_perception` tools, examples | L | A3-A7, A9 | Bench is the largest consumer |
| A14 | Tests: state machine (stop-before-start, double stop, FAILED), hook ordering/timeout/fail-closed, executor timeout returns on time, cancellation reaches tools, StateBased context not persisted; delete obsolete tests | M | each item | Write alongside items, not at the end |
| A15 | Docs + migration guide (old->new API table from section 3) | M | all | |

**Sequencing**:
- **Phase 1** (parallel, no dependencies between them): A1, A2, A5. Agree the A2 interface with the connector-side planner first.
- **Phase 2**: A3, then A4 and A10 in parallel.
- **Phase 3**: A6, then A7, A8, A11 and A12 in parallel.
- **Phase 4**: A9 (can start after A3 if the connector async API is ready) and A13.
- **Phase 5**: A15.

The critical path is A2, A3, A6, A9/A13. Total effort is roughly 5-7 weeks of agentic implementation with review, and A9 and A13 dominate.

## 5. Explicitly rejected (keep lean)

- A RAI-owned chat message hierarchy. LangChain `BaseMessage` stays inside agents; only tools, aggregators and HRI are LangChain-free.
- Deny-by-default tools, a capability or permission DSL, and policy files.
- LangChain `create_agent` and middleware as the core loop, and `langchain-mcp-adapters` (both bypass RAI hooks on the MCP and direct paths).
- LangGraph `interrupt()` for human-in-the-loop (use an awaiting hook).
- Persistent checkpointers or memory stores in core. The user can pass any LangGraph checkpointer.
- A multi-agent orchestration framework in core (megamind moves to bench).
- Agent restart after STOPPED. Background prefetch for aggregators.
- RAI code for LangSmith (native env vars) and generic OpenTelemetry (later, if asked).
- Entry-point tool discovery and plugin registries.
