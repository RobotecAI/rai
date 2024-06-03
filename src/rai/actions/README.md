# 📘 README for Action Modules

## 🚀 actions.py

### Overview

This module is the core of action management within the system. It defines a set of actions that can be executed within scenarios managed by the `ScenarioRunner`. Actions can range from sending alerts and emails to logging important system events.

### Currently Implemented

- **`MessageAdminAction`**: Logs critical messages, useful for debugging or alerting on crucial events.
- **`SoundAlarmAction`**: Activates an alarm system, could be extended to interface with physical or software-based alert systems.
- **`SendEmailAction`**: Uses an SMTP server to send an email, potentially with attached files like screenshots or logs.
- **`SendStopSignalAction`**: Sends a stop signal, generally used for stopping a process or operation safely.
- **`EventReportSaver`**: Saves a report about a particular event, capturing images, actions, and positional data to a log directory.

## 🤖 ros_actions.py

### Overview

Specifically tailored for actions that interact with the Robot Operating System (ROS), this file defines actions that directly communicate with ROS nodes and services to perform operations like stopping the robot or executing ROS commands.

### Currently Implemented

- **`StopRobotAction`**: Sends a command to stop the robot using a ROS service, crucial for emergency stops or controlled shutdowns.
- **`RosAPICallAction`**: Executes a ROS command, useful for configurations or operations that need to change dynamically based on scenario conditions.

## ⚙️ executor.py

### Overview

This module handles the execution of actions within the system. It provides the infrastructure to run actions either as simple direct calls or conditionally based on the scenario's state.

### Currently Implemented

- **`Executor`**: Executes an action in a separate thread, allowing the system to handle other tasks concurrently.
- **`ConditionalExecutor`**: Checks a condition before executing an action, ensuring actions are only run when appropriate, which helps in maintaining the system's integrity and relevance of actions.
