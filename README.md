# 🦊 RAI

**RAI** is a framework for creating conversations between users and assistants in the [ROS2](https://ros.org/) ecosystem. It uses predefined, flexible scenarios with built-in actions. The engine is designed to be adaptable and scalable, supporting a wide range of nodes across different domains.

## Planned demos 👀

- [agriculture demo 🌾](https://github.com/RobotecAI/rai-agriculture-demo)
- [husarion demo 🤖](https://github.com/RobotecAI/rai-husarion-demo)
- [manipulation demo 🦾](https://github.com/RobotecAI/rai-manipulation-demo)

## Table of Contents

- [Scenario Definition](#-scenario-definition)
  - [Scenario Building Blocks](#-scenario-building-blocks)
  - [Scenario Definition Example](#-scenario-definition-example)
- [Available LLM Vendors](#-available-llm-vendors)
  - [Vendors Initialization Examples](#-vendors-initialization-examples)
    - [Ollama](#ollama)
    - [OpenAI](#openai)
    - [AWS Bedrock](#aws-bedrock)
- [Integration with Robotic Systems](#-integration-with-robotic-systems)
- [Installation](#installation-instructions)
- [Further documentation](#further-documentation)

## General Architecture Diagram with Current and Planned Features

![rai_arch](docs/imgs/rai_arch.png)

## 🧩 Scenario Definition

A scenario is a programmatically defined sequence of interactions between a User and an Assistant (LLM). Each scenario consists of multiple components that dictate the flow of conversation and actions.

### 🏗️ Scenario Building Blocks

Scenarios can be built using the following elements:

- **Messages**: Static or dynamic content communicated to the user.
- **Conditional Scenarios**: Content that changes based on certain conditions.

For more about scenario building see: [docs/scenarios.md](docs/scenarios.md)\
For more about scenario running: [src/rai/scenario_engine](src/rai/scenario_engine)

#### For available tools see:

- 🔨 [Tools](./src/rai/tools/)
- 🤖 [ROS2 Actions](./src/rai/tools/ros/)

#### 📝 Scenario Definition Example

For example scenarios see:

- 🤖 [ROS2 scenario](./examples/husarion_poc_example.py)
- 🔄 [Simple scenario](./examples/agri_example.py)

## 🌐 Available LLM Vendors

We currently support the following vendors:

Locally hosted:

- 🏠 Ollama [link](https://ollama.com/)

Cloud hosted:

- ☁️ AWS Bedrock [link](https://aws.amazon.com/bedrock/)
- ☁️ OpenAI [link](https://platform.openai.com/)

Planned:

- ☁️ Anthropic [link](https://www.anthropic.com/api)
- ☁️ Cohere [link](https://cohere.com/)

For more see: [src/rai/vendors](src/rai/vendors)

### 🚀 Vendors Initialization Examples

#### Ollama

```python
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model='llava')
```

#### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

```python
from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
)
```

#### AWS Bedrock

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

## 🔗 Integration with Robotic Systems

This engine provides support for integration with robotic systems through ROS2, allowing for real-time control and feedback within various robotic applications.\
For more information see: [src/rai/communication](src/rai/communication)

## 📚 Installation and further documentation

### Installation instructions

#### Requirements

- python3.10^
- poetry

Additionally some of the modules or examples may require langfuse api keys for usage tracking. Contact repo mainteiners for api keys.

```bash
export LANGFUSE_PK="pk-lf-*****"
export LANGFUSE_SK="sk-lf-****"
export LANGFUSE_HOST=""
```

Poetry installation (probably other versions will work too):

```bash
python3 -m pip install poetry==1.8.3
```

#### Installation

1. Clone the repository:

```sh
git clone git@github.com:RobotecAI/rai-private.git
cd rai-private
```

2. Create and activate a virtual environment:

```sh
poetry install
poetry shell
```

##### Installation verification (optional)

1. Set vendor keys
2. Run pytest

```bash
pytest -m billable
```

> [!WARNING]
> Running the tests will trigger paid api calls.

3. Run example

This example mocks the interaction with [rai-agriculture-demo](https://github.com/RobotecAI/rai-agriculture-demo) by using static images and skipping ros2 communication. The full integration is planned in future releases.
```bash
pip install gdown
gdown --folder -O examples/imgs https://drive.google.com/drive/folders/1KRwCph465SBEMbuu5y1srzF9ZxVqjffw\?usp\=drive_link
python examples/agri_example.py
```

> [!NOTE]
> gdown is used for downloading images used in the agri_example.py

Expected outcome:

```
$ python examples/agri_example.py
2024-06-28 12:33:09 robo-pc-054 ScenarioRunner[2593946] INFO Starting conversation.
2024-06-28 12:33:09 robo-pc-054 langfuse[2593946] WARNING Item exceeds size limit (size: 4703784), dropping input/output of item.
2024-06-28 12:33:09 robo-pc-054 langfuse[2593946] WARNING Item exceeds size limit (size: 4703871), dropping input/output of item.
Running tool: StopTool with args: {}
Running tool: UseHonkTool with args: {}
2024-06-28 12:33:24 robo-pc-054 langfuse[2593946] WARNING Item exceeds size limit (size: 8150030), dropping input/output of item.
2024-06-28 12:33:24 robo-pc-054 langfuse[2593946] WARNING Item exceeds size limit (size: 8150117), dropping input/output of item.
Running tool: ContinueActionTool with args: {}
2024-06-28 12:33:43 robo-pc-054 ScenarioRunner[2593946] INFO Conversation completed.
2024-06-28 12:33:43 robo-pc-054 ScenarioRunner[2593946] INFO Conversation saved to: logs/ChatOpenAIxxxx-xx-xx_xx:xx:xx.xxxxxx/history.html
```

### Further documentation

For examples see [examples](examples/)\
For Message definition: [messages.md](docs/messages.md)\
For Scenario definition: [scenarios.md](docs/scenarios.md)

For more information see readmes in respective folders.

```
.
├── docs
│   ├── messages.md
│   └── scenarios.md
├── README.md
└── src
    └── rai
        ├── tools
        │   └── README.md
        ├── communication
        │   └── README.md
        ├── README.md
        └── scenario_engine
            └── README.md
```
