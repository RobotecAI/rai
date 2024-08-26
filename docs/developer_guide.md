# Developer Guide

RAI is a problem-agnostic framework designed to provide a general solution for various tasks. It is easily extendable, allowing developers to adapt and integrate new functionalities and tools to meet specific needs.

## Try out RAI

```python
from rai import ROS2Agent

agent = ROS2Agent(vendor='openai') # openai or bedrock
print(agent("What topics, services, and actions are available?"))
print(agent("Please describe the interfaces of two of the existing topics."))
print(agent("Please publish 'Hello RAI' to /chatter topic only once")) # make sure to listen first ros2 topic echo /chatter
```

## Adjusting RAI for Your Robot

### 1. Create Your Robot whoami configuration package [Optional]

The whoami package is used for configuring your robot within the RAI ecosystem. It helps define the robot's identity, self-understanding, ethical code, and documentation. For instructions see [configure RAI for your robot](create_robots_whoami.md).

> [!NOTE]  
> While RAI's agents can be started without a whoami configuration package, we strongly recommend integrating it for a more complete setup.

### 2. Implement new tools specific to your robot

To extend RAI with tools specific to your robot, implement new tool functions as shown below. These functions should be decorated with @tool and should interact with your robot's API.

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
    """Tool used for x-raying previously picked object"""
    return robot.scan_object()

# class based api, useful when tools use foreign objects like ROS 2 Node
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
    State retriever used for feeding state information to agent every interation.
    The state can consist of robot's logs, as well as any information that might
    be useful to the robot's operations and agent's reasoning.
    """
    return {"temperature": 30, "battery_state": 33, "logs": [...]}

```

### 3. Run the agent with new tools

Once you have implemented your tools, you can run the agent with these new tools as follows:

```python
from rai.agents.state_based import create_state_based_agent
from langchain_openai import ChatOpenAI
from myrobot import robot

llm = ChatOpenAI(model='gpt-4o') # initialize your vendor of choice
tools = [pick_up_object, scan_object, SayTool(robot=robot)]
agent = create_state_based_agent(llm=llm, state_retriever=state_retriever, tools=tools)
agent.invoke({"messages": ["Please pick up an object and scan it."]})
```

Additional resources:

- [Beta demos](demos.md).
- [Multimodal Messages](multimodal_messages.md) definition.
- Available ROS 2 packages: [ros packages](ros_packages.md).
- [Human-Robot Interface](human_robot_interface.md) through voice and text.
- [Manipulation](manipulation.md) with OpenVLA.
