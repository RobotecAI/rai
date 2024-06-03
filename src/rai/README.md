Directory tree

```
📂 Project Root
├── 🚀 actions
│   ├── 🧩 actions.py               # 🤖 Action Classes: Various robot and system interactions
│   ├── ⚙️ executor.py               # 🔄 Executors: Manage threaded action execution
│   └── 🤖 ros_actions.py           # 🛠️ ROS Actions: Specific methods for Robot Operating System interactions
├── 📡 communication
│   ├── 📶 communication.py         # 🌐 Standard Communication: Non-ROS implementations
│   └── 🤖 ros_communication.py     # 📡 ROS2 Communication: Interfaces for ROS2 systems
├── 📖 history_saver.py             # 📚 History Saver: Archive chat history in various formats
├── 📜 message.py                   # 📩 Message Structures: Define different types of chat messages
├── 🎭 prompts.py                   # 💡 Prompt Helpers: Generate and manage interactive chat prompts
├── 📏 requirements.py              # 📋 Requirements Check: Ensure message criteria are met
├── 🎬 scenario_engine
│   └── 🕹️ scenario_engine.py       # 🎮 Scenario Runner: Execute and control chat scenarios
└── 🤝 vendors
    └── 🔌 vendors.py               # 🌐 AI Vendors: Interface with external AI services

```

## Project Directory Structure Overview

### Directory and File Descriptions

#### 🚀 `actions`

- **`actions.py`**: Contains definitions of various `Action` classes that encapsulate specific tasks or operations. Implementations here can range from sending notifications to integrating with other software systems or APIs.
- **`executor.py`**: This file includes `Executor` and `ConditionalExecutor` classes for handling the execution of actions. These classes are crucial for ensuring actions are performed either unconditionally or based on specific conditions.
- **`ros_actions.py`**: Despite the robotics-oriented naming, this file could be repurposed to handle specific APIs or external system calls relevant to your application's context.

#### 📡 `communication`

- **`communication.py`**: Here, standard communication protocols or methods that aren't specific to any particular platform should be implemented. This could involve REST API communications, handling standard input/output, etc.
- **`ros_communication.py`**: Originally designed for ROS2 communications, this file can be adapted to handle interactions with other real-time systems or complex multi-component software environments.

#### 📖 `history_saver.py`

- Houses the `HistorySaver` class, which is responsible for archiving interaction histories in various formats like HTML, JSON, or Markdown. This class can be enhanced to include more sophisticated data handling or encryption for security purposes.

#### 📜 `message.py`

- Defines the structure of messages within the system. Custom message types can be created here to accommodate specific logging or data-passing needs between different parts of your application.

#### 🎭 `prompts.py`

- Contains helper functions and classes for generating and managing interactive prompts. This is useful for chatbots or other interactive systems where user input needs to be guided or restricted to specific responses.

#### 📏 `requirements.py`

- Implements `Requirement` classes that define various constraints on messages or actions, such as length limits or format validations. Extending this could involve adding custom validation rules based on new requirements.

#### 🎬 `scenario_engine`

- **`scenario_engine.py`**: Central to running predefined scenarios involving a series of actions and decisions. This file could be expanded with more complex decision-making capabilities or integrations with machine learning models for dynamic response generation.

#### 🤝 `vendors`

- **`vendors.py`**: Interfaces with different AI service providers. Modifications here could include adding new vendors or changing how responses are processed and handled, adapting to the specifics of different AI technologies or APIs.
