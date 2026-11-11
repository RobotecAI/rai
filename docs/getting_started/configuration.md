# Configuration

RAI reads its settings from `RAI_*` environment variables over an optional `rai.toml` file over built-in defaults. The file is the first of: a path passed in code, `$RAI_CONFIG`, `./rai.toml`. Secrets never go in the file: each provider reads its own environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `LANGFUSE_SECRET_KEY`, ...).

## rai.toml

<!-- test: file rai.toml -->

```toml
[models]
simple = "openai:gpt-4o-mini"         # fast, cheap tasks such as image descriptions
complex = "openai:gpt-4o"             # the agent's reasoning model
embeddings = "openai:text-embedding-3-small"
aliases = { vlm = "google_genai:gemini-3-flash", local = "ollama:llama3.1:8b" }

[models.provider_kwargs.openai]       # constructor kwargs per provider
temperature = 0

[tracing]
project = "my-robot"
langfuse = false                      # keys in LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
langsmith = false                     # key in LANGSMITH_API_KEY
```

Every key is optional. Unknown tables are ignored by `rai.settings`, so other parts of RAI keep their sections in the same file:

| Table        | Read by                                                               |
| ------------ | --------------------------------------------------------------------- |
| `[models]`   | `rai.llm.get_llm`, `get_embeddings` and every agent given a role name |
| `[tracing]`  | `rai.llm.get_tracing_callbacks`, attached to every `ReActAgent`       |
| `[mcp]`      | [`rai mcp serve` and the MCP client](../guides/mcp.md)                |
| `[manifest]` | [the robot manifest](../guides/manifest.md)                           |
| `[asr]`      | [speech recognition](../speech_to_speech/agents/asr.md) (rai_s2s)     |
| `[tts]`      | [text to speech](../speech_to_speech/agents/tts.md) (rai_s2s)         |

## Environment variables

Every setting has a `RAI_` variable; nested keys are joined with `__`. Variables win over the file:

```bash
export RAI_MODELS__COMPLEX="anthropic:claude-sonnet-5-5"
export RAI_TRACING__LANGFUSE=true
export RAI_CONFIG=/etc/rai/robot.toml    # use this file instead of ./rai.toml
```

The tables of other packages follow the same rule (`RAI_MCP__READ_ONLY=true`, `RAI_ASR__LANGUAGE=pl`).

## Models

A model is a `"provider:model"` string, the syntax of LangChain's `init_chat_model`. The text after the first colon is the model name, so `"ollama:llama3.1:70b"` works.

| Provider                   | Example                       | Extra       | Credentials                                                     |
| -------------------------- | ----------------------------- | ----------- | --------------------------------------------------------------- |
| `openai`                   | `openai:gpt-4o`               | `openai`    | `OPENAI_API_KEY`                                                |
| `anthropic`                | `anthropic:claude-sonnet-5-5` | `anthropic` | `ANTHROPIC_API_KEY`                                             |
| `google` (`google_genai`)  | `google:gemini-3-flash`       | `google`    | `GOOGLE_API_KEY`                                                |
| `aws` (`bedrock_converse`) | `aws:<Bedrock model id>`      | `aws`       | the AWS credential chain (`AWS_ACCESS_KEY_ID`, profiles, roles) |
| `ollama`                   | `ollama:llama3.1:8b`          | `ollama`    | none; a running `ollama serve`                                  |

Any other provider `init_chat_model` supports works once its LangChain package is installed.

To use an OpenAI-compatible server (vLLM, LM Studio, Ollama's `/v1` endpoint), point the `openai` provider at it:

```toml
[models]
complex = "openai:qwen3-32b"

[models.provider_kwargs.openai]
base_url = "http://localhost:8000/v1"
api_key = "unused"
```

Pick the best model you can afford for `complex`: tool calling and image understanding quality decide how well agents work. Local models cost nothing per call but need a GPU.

### In code

<!-- test: run -->

```python
from rai import get_llm, get_settings

settings = get_settings()
print(settings.models.complex)

llm = get_llm("simple")                         # a role
vlm = get_llm("vlm")                            # an alias from [models.aliases]
judge = get_llm("openai:gpt-4o", temperature=0) # a "provider:model" string and kwargs
```

`get_llm` returns a LangChain `BaseChatModel`. Constructor kwargs come from `models.provider_kwargs[provider]`, then from the call. `rai.llm.get_embeddings` does the same for embeddings models.

`get_settings()` is cached. Call `get_settings.cache_clear()` after changing the environment or the file, or use `rai.settings.load_settings(path)` to read a file without the cache.

### Configurator

The configurator is a Streamlit page that edits the `[models]` and `[tracing]` tables of `rai.toml` and checks that each model can be created. It needs the `ui` extra:

```bash
streamlit run "$(python -c 'import rai.ui.configurator as m; print(m.__file__)')"
```

## Tracing

Tracing records every model call and tool call of your agents. It is off by default.

-   **Langfuse** (open source, self-hostable): install the `langfuse` extra, set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`, and set `langfuse = true` (and `langfuse_host` for a self-hosted server) in `[tracing]`.
-   **LangSmith**: set `LANGSMITH_API_KEY` and `langsmith = true` (and `langsmith_host` if needed). `project` names the LangSmith project.

`ReActAgent` and `StateBasedAgent` attach the callbacks themselves. For direct model calls, pass them:

```python
from rai import get_llm, get_tracing_callbacks

get_llm("complex").invoke("Hello", config={"callbacks": get_tracing_callbacks()})
```

The [flight recorder](../guides/recorder.md) is the offline alternative: it writes every turn, tool call and hook decision to an MCAP file next to your ROS topics.
