# Local models

This page describes outcomes of open-source models testing with `rai`.

| Model                                     | Hosting                | GPU     | Notes                                                                        |
| ----------------------------------------- | ---------------------- | ------- | ---------------------------------------------------------------------------- |
| `qwen2.5:7b`                              | openai ollama endpoint | RTX4090 | - tool calling for [debugging_assistant][da]                                 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | vllm                   | RTX4090 | not useful for `rai`, since model has no tool calling                        |
| `Qwen/Qwen2.5-7B`                         | vllm                   | RTX4090 | tested in [debugging_assistant][da], looped when tried to call tool          |
| `openbmb/MiniCPM3-4B`                     | vllm                   | RTX4090 | works with [debugging_assistant][da] with [this setup][pcm_tool], not stable |

[da]: https://github.com/RobotecAI/rai/blob/development/docs/debugging_assistant.md
[pcm_tool]: https://github.com/vllm-project/vllm/issues/9692#issuecomment-2518948910
