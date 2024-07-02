# 🎬 ScenarioRunner Module

## Overview

The `ScenarioRunner` module is an essential part of our application. It enables running scenarios, which are series of messages and actions that simulate a conversation or a process. This is great for things like chatbots or automated systems that need to respond to user inputs in a consistent and logical way.

## Key Components

### 🏃 ScenarioRunner Class

- **Purpose**: Manages the execution of different scenarios. Each scenario is made up of parts that can be messages, actions, or decisions.
- **How it works**: Starts with a scenario, and runs through each part. It can send messages, execute actions, or make decisions based on certain conditions.
- **Caching**: Can remember previous responses to save time and resources. This is useful when responses are predictable and don't need to be recalculated.

## Saving and Logging

- **Saving**: Can save the entire conversation or scenario outcome as HTML. This is useful for keeping records or reviewing how scenarios unfold.
- **Logging**: Uses `coloredlogs` to make log messages easier to read. This is great for debugging and understanding the flow of scenarios.

## How to Use This Module

1. **Setup**: Make sure you have defined your scenarios using the available messages.
2. **Execution**: Use the `run()` method of the `ScenarioRunner` to start executing your scenario.
3. **Monitoring**: Keep an eye on the logs to understand how your scenario is processing and to catch any issues early.
4. **Review**: After running scenarios, you can review the saved files to analyze the outcomes and make improvements.

## Customizing Scenarios

- **Expand scenarios**: Add more message types or actions to handle new types of interactions.
