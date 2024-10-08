# Developer Guide

RAI is a flexible AI agent framework designed to handle tasks in generalized manner.
It is easily extendable, allowing developers to adapt and integrate new functionalities and tools to meet their needs.

## Development Environment Setup

1. Follow installation instructions from [Quick Start](../README.md#quick-start)
2. Activate [pre-commit](https://pre-commit.com) hook for automatic code checking
   and formatting (it uses [../.pre-commit-config.yaml](../.pre-commit-config.yaml)).

   ```bash
   pre-commit install
   ```

## Try out RAI

```python
from rai import ROS2Agent

agent = ROS2Agent() # vendor will be automatically initialized based on the config.toml
print(agent("What topics, services, and actions are available?"))
print(agent("Please describe the interfaces of two of the existing topics."))
print(agent("Please publish 'Hello RAI' to /chatter topic only once")) # make sure to listen first ros2 topic echo /chatter
```

## Adjusting RAI for Your Robot

### 1. Create Your Robot whoami configuration package

Follow instructions to [configure RAI identity for your robot](create_robots_whoami.md).

### 2. Test Your Robot Using the Text HMI

To run the fully initialized Streamlit HMI, use the following command, replacing `my_robot_whoami` with the name of the package you created:

```bash
streamlit run src/rai_hmi/rai_hmi/text_hmi.py my_robot_whoami
```

To verify if Streamlit successfully loaded the configuration:

1. Expand the "System status" menu in the Streamlit interface.

2. Check for the following indicators:
   - "robot_database": true
   - "system_prompt": true

If both indicators are present and set to true, your configuration has been loaded correctly. You can now interact with your robot by:

1. Asking about its identity and purpose

2. Inquiring about its capabilities

3. Requesting information on the ROS topics it can access

This interaction will help you verify that the robot's 'whoami' package is functioning as intended and providing accurate information about your robot's configuration.

### 3. Implement new tools specific to your robot

RAI has general capabilities to interact through ROS interfaces such as actions and topics.
However, you can extend RAI with tools dedicated to what your robot needs to do.
These functions should be decorated with @tool and should interact with your robot's API.
See the example below.

```python
from langchain.tools import tool, BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Type
from myrobot import robot

# decorator based api, for basic tools
@tool
def pick_up_object(bbox_3d):
    """Tool used for picking up objects"""
    return robot.pick_up_object(bbox_3d)

@tool
def scan_object():
    """Tool used for examining previously picked object"""
    return robot.scan_object()

# class based api, useful when tools use objects like ROS 2 Node
class SayToolInput(BaseModel):
    text: str = Field(..., description="Text to be said.")

class SayTool(BaseTool):
    name: str = "say"
    description: str = "Tool used for speaking in e.g. human-robot conversation"
    robot: type[Robot]
    args_schema: Type[SayToolInput] = SayToolInput

    def _run(self, text: str):
        return self.robot.speak(text)

def state_retriever():
    """
    State retriever used for feeding state information to agent every iteration.
    The state can consist of robot's logs, as well as any information that might
    be useful to the robot's operations and agent's reasoning.
    """
    return {"temperature": 30, "battery_state": 33, "logs": [...]}

```

### 4. Run the agent with new tools

Once you have implemented your tools, you can run the agent with these new tools as follows:

```python
from rai.agents.state_based import create_state_based_agent
from rai.utils.model_initialization import get_llm_model
from myrobot import robot

llm = get_llm_model(model_type='complex_model') # initialize your vendor of choice in config.toml
tools = [pick_up_object, scan_object, SayTool(robot=robot)]
agent = create_state_based_agent(llm=llm, state_retriever=state_retriever, tools=tools)
agent.invoke({"messages": ["Please pick up an object and scan it."]})
```

Additional resources:

- [Tracing](tracing.md) configuration for genAI models and agents.
- [Beta demos](demos.md).
- [Multimodal Messages](multimodal_messages.md) definition.
- Available ROS 2 packages: [ros packages](ros_packages.md).
- [Human-Robot Interface](human_robot_interface.md) through voice and text.
- [Manipulation](manipulation.md) with Grounded SAM 2.

## Architecture diagram

![rai_arch.png](imgs%2Frai_arch.png)
