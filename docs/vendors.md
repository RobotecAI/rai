# Vendors Initialization Examples

## Ollama

```python
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model='llava')
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
