# Tool use and development

Tools are a fundamental concept in LangChain that allow AI models to interact with external systems
and perform specific operations. Think of tools as callable functions that bridge the gap between
natural language understanding and system execution.

RAI offers a comprehensive set of pre-built tools, including both general-purpose and ROS 2-specific
tools [here](https://github.com/RobotecAI/rai/tree/{{branch}}/src/rai_core/rai/tools/ros2). However, in some cases, you may need to develop
custom tools tailored to specific robots or applications. This guide demonstrates how to create
custom tools in RAI using the [LangChain framework](https://python.langchain.com/docs/).

## How LLMs Understand Tools

When an LLM is given access to tools, it needs to understand:

1. What tools are available
2. What each tool does
3. What inputs each tool accepts

This understanding is achieved through the tool's metadata fields, which are automatically processed by LangChain and made available to the LLM. The LLM uses this information to:

-   Determine which tool to use for a given task
-   Format the correct input arguments
-   Interpret the tool's output

The key fields that enable this understanding are:

-   `name`: A unique identifier for the tool, does not have to be equal to the function/class name
-   `description`: A natural language explanation of what the tool does
-   `args_schema`: A structured definition of the tool's input parameters

RAI supports two primary approaches for implementing tools, each with distinct advantages:

### `BaseTool` Class

-   Offers full control over tool behavior and lifecycle
-   Allows configuration parameters
-   Supports stateful operations (e.g., maintaining ROS 2 connector instances)

### `@tool` Decorator

-   Provides a lightweight, functional approach
-   Ideal for stateless operations
-   Minimizes boilerplate code
-   Suited for simple, single-purpose tools

Use the `BaseTool` class when state management, or extensive configuration is required. Choose the
`@tool` decorator for simple, stateless functionality where conciseness is preferred.

---

## Creating a Custom Tool

LangChain tools typically return either a string or a tuple containing a string and an artifact.

RAI extends LangChain's tool capabilities by supporting **multimodal tools**—tools that return not
only text but also other content types, such as images, audio, or structured data. This is achieved
using a special object called `MultimodalArtifact` along with a custom `ToolRunner` class.

---

### Single-Modal Tool (Text Output)

Here's an example of a single-modal tool implemented using class inheritance:

!!! tip "Class-based tools"

    Class-based tools provide more control and flexibility compared to function-based tools:

    - Allow passing additional parameters (e.g., connectors, configuration)
    - Support stateful operations

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type


class GrabObjectToolInput(BaseModel):
    """Input schema for the GrabObjectTool.

    This schema defines the expected input parameters for the tool.
    The Field class provides additional metadata that helps the LLM understand
    the parameter's purpose and constraints.
    Fields must include a type annotation.
    """
    object_name: str = Field(description="The name of the object to grab")


class GrabObjectTool(BaseTool):
    """Tool for grabbing objects using a robot.

    The following fields are crucial for LLM understanding:

    name: A unique identifier that the LLM uses to reference this tool
    description: A natural language explanation that helps the LLM understand when to use this tool
    args_schema: Links to the input schema, helping the LLM understand required parameters
    """
    name: str = "grab_object"
    description: str = "Grabs a specified object using the robot's manipulator"
    args_schema: Type[GrabObjectToolInput] = GrabObjectToolInput

    robot_arm: RobotArm # custom parameter for dependency injection

    def _run(self, object_name: str) -> str:
        """Execute the object grabbing operation.

        The LLM will receive this output and can use it to:
        1. Determine if the operation was successful
        2. Extract relevant information
        3. Decide on next steps
        """
        try:
            status = self.robot_arm.grab_object(object_name)
            return f"Successfully grabbed object: {object_name}, status: {status}"
        except Exception as e:
            return f"Failed to grab object: {object_name}, error: {str(e)}"
```

Alternatively, using the `@tool` decorator for simpler, stateless operations:

```python
from langchain_core.tools import tool

@tool
def grab_object(object_name: str) -> str:  # function name is equal to name in class API
    """Grabs a specified object using the robot's manipulator."""  # equal to description in class API
    try:
        status = robot_arm.grab_object(object_name)
        return f"Successfully grabbed object: {object_name}, status: {status}"
    except Exception as e:
        return f"Failed to grab object: {object_name}, error: {str(e)}"
```

---

### Multimodal Tool (Text + Image Output)

RAI supports multimodal tools through the `rai.agents.langchain.core.ToolRunner` class. These tools
must use this runner either directly or via agents such as [create_react_runnable](https://github.com/RobotecAI/rai/blob/{{branch}}/src/rai_core/rai/agents/langchain/runnables.py#L86) to handle multimedia output correctly.

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Tuple
from rai.messages import MultimodalArtifact


class Get360ImageToolInput(BaseModel):
    """Input schema for the Get360ImageTool."""
    topic: str = Field(description="The topic name for the 360 image")


class Get360ImageTool(BaseTool):
    """Tool for retrieving 360-degree images."""
    name: str = "get_360_image"
    description: str = "Retrieves a 360-degree image from the specified topic"
    args_schema: Type[Get360ImageToolInput] = Get360ImageToolInput
    response_format: str = "content_and_artifact"

    def _run(self, topic: str) -> Tuple[str, MultimodalArtifact]:
        try:
            image = robot.get_360_image(topic)
            return "Successfully retrieved 360 image", MultimodalArtifact(images=[image])
        except Exception as e:
            return f"Failed to retrieve image: {str(e)}", MultimodalArtifact(images=[])
```

---

### ROS 2 Tools

RAI includes a base class for ROS 2 tools, supporting configuration of readable, writable, and
forbidden topics/actions/services, as well as [ROS 2 connector](../API_documentation/connectors/ROS_2_Connectors.md).

```python
from rai.tools.ros2.base import BaseROS2Tool
from pydantic import BaseModel, Field
from typing import Type, cast
from sensor_msgs.msg import PointCloud2


class GetROS2LidarDataToolInput(BaseModel):
    """Input schema for the GetROS2LidarDataTool."""
    topic: str = Field(description="The topic name for the LiDAR data")


class GetROS2LidarDataTool(BaseROS2Tool):
    """Tool for retrieving and processing LiDAR data."""
    name: str = "get_ros2_lidar_data"
    description: str = "Retrieves and processes LiDAR data from the specified topic"
    args_schema: Type[GetROS2LidarDataToolInput] = GetROS2LidarDataToolInput

    def _run(self, topic: str) -> str:
        try:
            lidar_data = self.connector.receive_message(topic)
            msg = cast(PointCloud2, lidar_data.payload)
            # Process the LiDAR data
            return f"Successfully processed LiDAR data. Detected objects: ..."
        except Exception as e:
            return f"Failed to process LiDAR data: {str(e)}"
```

??? note "For more information on BaseROS2Tool refer to the source code"

    ::: rai.tools.ros2.base.BaseROS2Tool

---

## Tool Initialization

Tools can be initialized with parameters such as a connector, enabling custom configurations for ROS
2 environments.

```python
from rai.communication.ros2 import ROS2Connector
from rai.tools.ros2 import (
    GetROS2ImageTool,
    GetROS2TopicsNamesAndTypesTool,
    PublishROS2MessageTool,
)

def initialize_tools(connector: ROS2Connector):
    """Initialize and configure ROS 2 tools.

    Returns:
        list: A list of configured tools.
    """
    readable_names = ["/color_image5", "/depth_image5", "/color_camera_info5"]
    forbidden_names = ["cmd_vel"]
    writable_names = ["/to_human"]

    return [
        GetROS2ImageTool(
            connector=connector, readable=readable_names, forbidden=forbidden_names
        ),
        GetROS2TopicsNamesAndTypesTool(
            connector=connector,
            readable=readable_names,
            forbidden=forbidden_names,
            writable=writable_names,
        ),
        PublishROS2MessageTool(
            connector=connector, writable=writable_names, forbidden=forbidden_names
        ),
    ]
```

---

### Using Tools in a RAI Agent (Distributed Setup)

```python
from rai.agents import ReActAgent
from rai.communication import ROS2Connector, ROS2HRIConnector
from rai.tools.ros2 import ROS2Toolkit
from rai.communication.ros2 import ROS2Context
from rai import AgentRunner

@ROS2Context()
def main() -> None:
    """Initialize and run the RAI agent with configured tools."""
    connector = ROS2HRIConnector()
    ros2_connector = ROS2Connector()
    agent = ReActAgent(
        connectors={"/to_human": connector},
        tools=initialize_tools(connector=ros2_connector),
    )
    runner = AgentRunner([agent])
    runner.run_and_wait_for_shutdown()

# Example:
# ros2 topic pub /from_human rai_interfaces/msg/HRIMessage "{\"text\": \"What do you see?\"}"
# ros2 topic echo /to_human rai_interfaces/msg/HRIMessage
```

---

### Using Tools in LangChain/LangGraph Agent (Local Setup)

```python
from rai.agents.langchain import create_react_runnable
from langchain.schema import HumanMessage
from rai.communication.ros2 import ROS2Context

@ROS2Context()
def main():
    ros2_connector = ROS2Connector()
    agent = create_react_runnable(
        tools=initialize_tools(connector=ros2_connector),
        system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
    )
    state = {'messages': []}
    while True:
        input_text = input("Enter a prompt: ")
        state['messages'].append(HumanMessage(content=input_text))
        response = agent.invoke(state)
        print(response)
```

---

## Related Topics

-   [Connectors](../API_documentation/connectors/overview.md)
-   [ROS2Connector](../API_documentation/connectors/ROS_2_Connectors.md#ros2connector)
-   [ROS2HRIConnector](../API_documentation/connectors/ROS_2_Connectors.md#ros2hriconnector)
