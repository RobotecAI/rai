# RAI Whoami

![RAI Whoami](./docs/images/rai_whoami.png)

## Overview

**RAI Whoami** is a Python package designed to extract and synthesize robot embodiment information from a structured directory of documentation, images, and URDFs.  
It generates a comprehensive system prompt (embodiment info) for robots controlled by LLMs, enabling advanced reasoning guided by the robot's embodiment setup.

---

## How It Works

Given a directory containing robot documentation (documents, images, URDFs), RAI Whoami processes these resources to produce a structured representation of the robot, including:

- **Rules**: Extracted operational or safety rules.
- **Behaviors**: Descriptions of robot behaviors.
- **Capabilities**: Functional and physical capabilities.
- **Images**: Visual representations.
- **Vector Database**: Embeddings of the robot's documentation. (optional)

This embodiment info is then used to create a system prompt for LLM-based agents, enabling them to reason about and interact with the robot effectively.

### Directory Structure

Prepare your robot documentation directory as follows:

```
documentation_dir/
├── images/          # png, jpg, jpeg files
├── documentation/   # pdf, docx, doc, md files
├── urdf/            # urdf files
```

### Building the Embodiment Info

To generate the system prompt from your documentation directory:

```bash
python src/rai_whoami/rai_whoami/build_whoami.py documentation_dir [--output_dir output_dir] [--build-vector-db]
```

---

## Using with ROS2 and ReActAgent

Integrate the generated embodiment info into your LLM-powered robot agent:

```python
from rai_whoami import EmbodimentInfo
from rai.agents import ReActAgent, wait_for_shutdown
from rai.communication.ros2 import ROS2HRIConnector

info = EmbodimentInfo.from_directory("path/to/documentation_dir/")
system_prompt = info.to_langchain()  # Convert EmbodimentInfo to a system prompt

connector = ROS2HRIConnector()
agent = ReActAgent(
    target_connectors={"ros2": connector},
    system_prompt=system_prompt,
)

agent.run()
connector.register_callback("/from_human", agent)
wait_for_shutdown([agent])
```
