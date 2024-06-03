# Scenario building

Scenario can consist of:

```python
ScenarioPartType = Union[
    ConstantMessage, # UserMessage, SystemMessage
    AssistantMessage,
    ConditionalMessage,
    ConditionalExecutor,
    ConditionalScenario,
    Executor,
]
```

Example scenario:

```python
from rai.message import (
    Message,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ConditionalMessage,
)
from rai.actions.executor import ConditionalExecutor, Executor
from rai.actions.actions import SendEmailAction

scenario = [
    SystemMessage(
        content="You are an AI assistant specialized in providing technical support for autonomous tractors.",
    ),
    UserMessage(
        content="Analyze the surroundings and determine if the path is clear. Reply with just one word: yes or no.",
        images=[Message.preprocess_image('examples/imgs/tractor_view.png')],
    ),
    AssistantMessage(),
    ConditionalScenario(
        if_true=[
            UserMessage(content="The path seems clear. Please proceed with the current path."),
            AssistantMessage(),
        ],
        if_false=[
            UserMessage(content="There seems to be an obstacle. Do you need more information?"),
            AssistantMessage(),
        ],
        condition=lambda x: "yes" in x[-1]["content"].lower(),
    ),
    ConditionalExecutor(
        action=SendEmailAction(email="support@example.com"),
        condition=lambda x: "yes" in x[-1]["content"].lower(),
    ),
    Executor(
        action=SendEmailAction(email="admin@example.com"),
    ),
]

```
