# Vendor setup

RAI supports multiple vendors for AI models and tracing tools, both open-source and
commercial APIs. To setu[ it is recommended to use the [RAI Configurator][configurator].

Alternatively vendors can be configured manually in `config.toml` file.

## Vendors Overview

The table summarizes vendor alternative for core AI service and optional RAI modules:

| Module                                          | Open source        | Alternative             | Why to consider alternative?                                             | More information                                                                                                                                                                 |
| ----------------------------------------------- | ------------------ | ----------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [LLM service](#llm-model-configuration-in-rai)  | Ollama             | OpenAI, Bedrock         | Overall performance of the LLM models, supported modalities and features | [LangChain models](https://docs.langchain4j.dev/integrations/language-models/)                                                                                                   |
| **Optional:** [Tracing tool](./tracing.md)      | Langfuse           | LangSmith               | Better integration with LangChain                                        | [Comparison](https://langfuse.com/faq/all/langsmith-alternative)                                                                                                                 |
| **Optional:** [Text to speech](#text-to-speech) | KokoroTTS, OpenTTS | ElevenLabs              | Arguably, significantly better voice synthesis                           | <li> [KokoroTTS](https://huggingface.co/hexgrad/Kokoro-82M#usage) </li><li> [OpenTTS GitHub](https://github.com/synesthesiam/opentts) </li><li> [RAI voice interface][s2s] </li> |
| **Optional:** [Speech to text](#speech-to-text) | Whisper            | OpenAI Whisper (hosted) | When suitable local GPU is not an option                                 | <li> [Whisper GitHub](https://github.com/openai/whisper) </li><li> [RAI voice interface][s2s] </li>                                                                              |

> [!TIP] Best-performing AI models
>
> Our recommendation, if your environment allows it, is to go with _OpenAI_ _GPT4o_ model,
> _ElevenLabs_ for TTS, locally-hosted _Whisper_, and Langsmith.

## LLM Model Configuration in RAI

In RAI you can configure 2 models: `simple model` and `complex model`:

-   `complex model` should be used for sophisticated tasks like multi-step reasoning.
-   `simple model` is more suitable for simpler tasks for example image description.

```python
from rai import get_llm_model

complex_llm = get_llm_model(model_type="complex")
simple_llm = get_llm_model(model_type="simple")
```

## Vendors Installation

### Ollama

Ollama can be used to host models locally.

1. Install `Ollama` see: [https://ollama.com/download](https://ollama.com/download)
2. Start Ollama server: `ollama serve`
3. Choose LLM model and endpoint type. Ollama server deliveres 2 endpoints:
    - Ollama endpoint: [RAI Configurator][configurator] -> `Model Selection` -> `ollama` vendor
    - OpenAI endpoint: [RAI Configurator][configurator] -> `Model Selection` -> `openai` vendor -> `Use OpenAI compatible API`
      Both endpoints should work interchangeably and decision is only dedicated by user's convenience.

### OpenAI

1. Setup your [OpenAI account](https://platform.openai.com/docs/overview), generate
   and set the API key:
   `bash export OPENAI_API_KEY="sk-..." `
2. Use [RAI Configurator][configurator] -> `Model Selection` -> `ollama` vendor

### AWS Bedrock

1. Set AWS Access Keys keys to your AWS account.

    ```bash
    export AWS_ACCESS_KEY_ID="..."
    export AWS_SECRET_ACCESS_KEY="..."
    export AWS_SESSION_TOKEN="..."
    ```

2. Use [RAI Configurator][configurator] -> `Model Selection` -> `bedrock` vendor

## Complex LLM Model Configuration

For custom setups please use LangChain API.

```python
from langchain_openai.chat_models import ChatOpenAI
from langchain_aws.chat_models import ChatBedrock
from langchain_community.chat_models import ChatOllama

llm1 = ChatOpenAI(model="gpt-4o")
llm2 = ChatOllama(model='llava')
llm = ChatBedrock(model="anthropic.claude-3-opus-20240229-v1:0")
```

## Text To Speech

For configuration use `Text To Speech` tab in [RAI Configurator][configurator].

Usage examples can be found in [Voice Interface Tutorial][s2s]

## Speech To Text

For configuration use `Speech Recognition` tab in [RAI Configurator][configurator].

Usage examples can be found in [Voice Interface Tutorial][s2s]

[configurator]: ./install.md#15-configure-rai
[s2s]: ../tutorials/voice_interface.md
