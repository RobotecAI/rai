# ROS 2 Debugging Assistant

The ROS 2 Debugging Assistant is an interactive tool that helps developers inspect and troubleshoot their ROS 2 systems using natural language. It provides a chat-like interface powered by Streamlit where you can ask questions about your ROS 2 setup and execute common debugging commands.

## Features

- Interactive chat interface for debugging ROS 2 systems
- Real-time streaming of responses and tool executions
- Support for common ROS 2 debugging commands:
  - `ros2 topic`: topic inspection and manipulation
  - `ros2 service`: service inspection and calling
  - `ros2 node`: node information
  - `ros2 action`: action server details and goal sending
  - `ros2 interface`: interface inspection
  - `ros2 param`: checking and setting parameters

## Running the Assistant

1. Make sure you have RAI installed and configured according to the [setup instructions](../README.md#setup)

2. Launch the debugging assistant:

```sh
source setup_shell.sh
streamlit run examples/debugging_assistant.py
```

## Usage Examples

Here are some example queries you can try:

- "What topics are currently available?"
- "Show me the message type for /cmd_vel"
- "List all active nodes"
- "What services does the /robot_state_publisher node provide?"
- "Show me information about the /navigate_to_pose action"

## How it Works

The debugging assistant uses RAI's conversational agent capabilities combined with ROS 2 debugging tools. The key components are:

1. **Streamlit Interface**: Provides the chat UI and displays tool execution results
2. **ROS 2 Tools**: Collection of debugging tools that wrap common ROS 2 CLI commands
3. **Streaming Callbacks**: Real-time updates of LLM responses and tool executions

## Limitations

- The assistant can only execute safe, read-only commands by default
- Some complex debugging scenarios may require manual intervention
- Performance depends on the chosen LLM vendor and model
