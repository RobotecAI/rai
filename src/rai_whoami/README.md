# RAI Whoami

## Description

The RAI Whoami is a package providing tools to extract information about a robot from a given directory.
Including pdf, docx, doc, md, urdf files and images.

## Creating a whoami configuration for a robot

### Prerequisites

- A directory with the following structure:

```
documentation_dir/
├── images/ # png, jpg, jpeg files
├── documentation/ # pdf, docx, doc, md files
├── urdf/ # urdf files
```

### Building the whoami configuration

```bash
python src/rai_whoami/rai_whoami/build_whoami.py documentation_dir --output_dir output_dir
```

### Using the whoami configuration within ROS2 and ReActAgent

```python
from rai_whoami import EmbodimentInfo
from rai.agents import ReActAgent
from rai.communication.ros2 import ROS2HRIConnector
from rai.agents import wait_for_shutdown

info = EmbodimentInfo.from_directory("path/to/documentation_dir/")
system_prompt = info.to_langchain()  # Convert EmbodimentInfo to a system prompt from the robot documentation

connector = ROS2HRIConnector()  # Create a connector for ROS2-based human-robot interaction
agent = ReActAgent(
    target_connectors={"ros2": connector},  # Use the ROS2 connector for agent communication
    system_prompt=system_prompt,
)

agent.run()
connector.register_callback("/from_human", agent)  # Handle incoming human messages from the specified ROS2 topic
wait_for_shutdown([agent])  # Block until the agent is shut down
```
