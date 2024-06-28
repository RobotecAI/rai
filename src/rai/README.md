Directory tree

```
📂 rai
├── communication
│   ├── communication.py         # Non-ROS communication implementations
│   ├── ros_communication.py     # Interfaces for ROS2 systems
├── history_saver.py             # Archive chat history in various formats
├── scenario_engine
│   ├── messages.py              # Message structures specific to scenario engine
│   ├── scenario_engine.py       # Scenario Runner: Execute and control chat scenarios
│   └── tool_runner.py           # Manage tool execution for scenarios
└── tools
    ├── hmi_tools.py             # Human-Machine Interface utilities
    ├── planning_tools.py        # Tools for planning and scheduling
    └── ros
        ├── cat_demo_tools.py    # Tools for demonstration purposes
        ├── cli_tools.py         # Command Line Interface ros utilities
        ├── mock_tools.py        # Mock implementations for testing
        └── tools.py             # General tools for ROS
```

## Project Directory Structure Overview

### Directory and File Descriptions

#### 🚀 `tools`

- **`tools.ros`**: ROS oriented tools
- **`tools.planning_tools`**: Planning oriented tools
- **`hmi_tools.py`**: Tools focused on providing seamless Human-Machine interface

#### 📡 `communication`

- **`communication.py`**: Here, standard communication protocols or methods that aren't specific to any particular platform should be implemented. This could involve REST API communications, handling standard input/output, etc.
- **`ros_communication.py`**: ROS 2 communications utils.

#### 📖 `history_saver.py`

- Houses the `HistorySaver` class, which is responsible for archiving interaction histories in HTML format.

#### 🎬 `scenario_engine`

- **`scenario_engine.py`**: Central to running predefined scenarios involving a series of actions and decisions. This file could be expanded with more complex decision-making capabilities or integrations with machine learning models for dynamic response generation.
