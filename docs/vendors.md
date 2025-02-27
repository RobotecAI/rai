# Vendors Initialization Examples

## Ollama

For installation see: https://ollama.com/. Then start
[ollama server](https://github.com/ollama/ollama?tab=readme-ov-file#start-ollama) with
`ollama serve` command.

```python
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model='llava')
```

### Configure ollama wth OpenAI compatible API

Ollama supports OpenAI compatible APIs (see [details](https://ollama.com/blog/openai-compatibility)).

> [!TIP]
> Such a setup might be more convenient if you frequently switch between OpenAI API and
> local models.

To configure ollama through OpenAI API in `rai`:

1. Add `base_url` to [config.toml](../config.toml)

```toml
[openai]
simple_model = "llama3.2"
complex_model = "llama3.2"
...
base_url = "http://localhost:11434/v1"
```

### Example of setting up vision models with tool calling

In this example `llama3.2-vision` will be used.

1. Create a custom ollama `Modelfile` and load the model

> [!NOTE]
> Such setup is not officially supported by Ollama and it's not guaranteed to be
> working in all cases.

```shell
ollama pull llama3.2
echo FROM llama3.2-vision > Modelfile
echo 'TEMPLATE """'"$(ollama show --template llama3.2)"'"""' >> Modelfile
ollama create llama3.2-vision-tools
```

3. Configure the model through an OpenAI compatible API in [config.toml](../config.toml)

```toml
[openai]
simple_model = "llama3.2-vision-tools"
complex_model = "llama3.2-vision-tools"
...
base_url = "http://localhost:11434/v1"
```

## OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

```python
from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
)
```

## AWS Bedrock

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."
```

```python
from langchain_aws.chat_models import ChatBedrock

llm = ChatBedrock(
    model="anthropic.claude-3-opus-20240229-v1:0",
)
```
