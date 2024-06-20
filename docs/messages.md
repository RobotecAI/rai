# Message Types

RAI implements MultimodalMessage that allows using image and audio* information.\
*audio is currently added as a placeholder

## Usage

Use multimodal message via it's concrete implementation.

```python
class HumanMultimodalMessage(HumanMessage, MultimodalMessage):
class SystemMultimodalMessage(SystemMessage, MultimodalMessage):
class AiMultimodalMessage(AIMessage, MultimodalMessage):
class ToolMultimodalMessage(ToolMessage, MultimodalMessage):
```

Example:

```python
from rai.scenario_engine.messages import HumanMultimodalMessage, preprocess_image
from langchain_openai.chat_models import ChatOpenAI

base64_image = preprocess_image('https://raw.githubusercontent.com/RobotecAI/RobotecGPULidar/develop/docs/image/rgl-logo.png')

llm = ChatOpenAI(model="gpt-4o")
msg = [HumanMultimodalMessage(content='This is an example', images=[base64_image])]
llm.invoke(msg)
```

Implementation of the following messages is identical: HumanMultimodalMessage, SystemMultimodalMessage, AiMultimodalMessage.

ToolMultimodalMessage has an addition of postprocess method, which converts the ToolMultimodalMessage into format that is compatible with a chosen vendor.
