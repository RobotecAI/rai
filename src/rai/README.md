Directory tree

```
📂 rai
├── 📡 communication
│   ├── 📶 communication.py         # 🌐 Standard Communication: Non-ROS implementations
│   ├── 🤖 ros_communication.py     # 📡 ROS2 Communication: Interfaces for ROS2 systems
│   └── 📄 README.md                # 📝 Documentation for communication module
├── 📖 history_saver.py             # 📚 History Saver: Archive chat history in various formats
├── 🎬 scenario_engine
│   ├── 📜 messages.py              # 💬 Message structures specific to scenario engine
│   ├── 🕹️ scenario_engine.py       # 🎮 Scenario Runner: Execute and control chat scenarios
│   └── 🛠️ tool_runner.py           # 🛠️ Tool Runner: Manage tool execution for scenarios
└── 🔧 tools
    ├── 📟 hmi_tools.py             # 🛠️ HMI Tools: Human-Machine Interface utilities
    ├── 📐 planning_tools.py        # 📊 Planning Tools: Tools for planning and scheduling
    └── 🤖 ros
        ├── 🐱 cat_demo_tools.py    # 🐱 Cat Demo Tools: Tools for demonstration purposes
        ├── 🖥️ cli_tools.py         # 💻 CLI Tools: Command Line Interface ros utilities
        ├── 🧩 mock_tools.py        # 🎭 Mock Tools: Mock implementations for testing
        └── 🔧 tools.py             # 🛠️ General tools for ROS
```

## Project Directory Structure Overview

### Directory and File Descriptions

#### 🚀 `tools`

- **`ros`**: ROS oriented tools
- **`other`**: Standard tools

#### 📡 `communication`

- **`communication.py`**: Here, standard communication protocols or methods that aren't specific to any particular platform should be implemented. This could involve REST API communications, handling standard input/output, etc.
- **`ros_communication.py`**: Originally designed for ROS2 communications, this file can be adapted to handle interactions with other real-time systems or complex multi-component software environments.

#### 📖 `history_saver.py`

- Houses the `HistorySaver` class, which is responsible for archiving interaction histories in HTML format.

#### 🎬 `scenario_engine`

- **`scenario_engine.py`**: Central to running predefined scenarios involving a series of actions and decisions. This file could be expanded with more complex decision-making capabilities or integrations with machine learning models for dynamic response generation.
