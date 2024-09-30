# Vendors Overview

The table summarizes vendor alternative for core AI service and optional RAI modules: 

| Module                       | Open source | Alternative             | Why to consider alternative?                                           | More information                                                                                                              |
|------------------------------| ----------- | ----------------------- |------------------------------------------------------------------------| ----------------------------------------------------------------------------------------------------------------------------- |
| AI service                   | Ollama      | OpenAI, Bedrock         | Overall performance of the AI model, supported modalities and features | [LangChain models](https://docs.langchain4j.dev/integrations/language-models/)                                                |
| __Optional:__ Tracing tool   | Langfuse    | LangSmith               | Better integration with LangChain                                      | [Comparison](https://langfuse.com/faq/all/langsmith-alternative)                                                              |
| __Optional:__ Text to speech | OpenTTS     | ElevenLabs              | Arguably, significantly better voice synthesis                         | <li> [OpenTTS GitHub](https://github.com/synesthesiam/opentts) </li><li> [RAI voice interface](docs/voice_interface.md) </li> |
| __Optional:__ Speech to text | Whisper     | OpenAI Whisper (hosted) | When suitable local GPU is not an option                               | <li> [Whisper GitHub](https://github.com/openai/whisper) </li><li> [RAI voice interface](docs/voice_interface.md) </li>       |
