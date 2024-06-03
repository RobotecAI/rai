# 🎬 ScenarioRunner Module

## Overview

The `ScenarioRunner` module is an essential part of our application. It lets us run scenarios, which are series of messages and actions that simulate a conversation or a process. This is great for things like chatbots or automated systems that need to respond to user inputs in a consistent and logical way.

## Key Components

### 🏃 ScenarioRunner Class

- **Purpose**: Manages the execution of different scenarios. Each scenario is made up of parts that can be messages, actions, or decisions.
- **How it works**: Starts with a scenario, and runs through each part. It can send messages, execute actions, or make decisions based on certain conditions.
- **Caching**: Can remember previous responses to save time and resources. This is handy when responses are predictable and don't need to be recalculated.

### 🌿 ConditionalScenario Class

- **Purpose**: Allows branching in scenarios. This means you can have different paths depending on certain conditions.
- **How it works**: Checks a condition and chooses a path accordingly. This helps in creating flexible and dynamic scenarios that can adapt to different situations.

## Functions and Methods

- **`run()`**: Starts the scenario and manages its execution. Keeps track of all messages and actions, ensuring everything is done in order.
- **`_run()`**: The heart of the module, where the scenario is actively processed. It checks the type of each part (message or action) and handles it accordingly.
- **`_handle_assistant_message()`**: Deals with messages from the assistant, ensuring they meet certain requirements and deciding when to retry sending a message if it doesn’t meet these requirements.
- **`_handle_conditional_message()`**: Manages messages that depend on certain conditions, choosing the right message based on the scenario's current state.

## Saving and Logging

- **Saving**: Can save the entire conversation or scenario outcome as HTML or Markdown. This is useful for keeping records or reviewing how scenarios unfold.
- **Logging**: Uses `coloredlogs` to make log messages easier to read. This is great for debugging and understanding the flow of scenarios.

## How to Use This Module

1. **Setup**: Make sure you have defined your scenarios using the available message and action types.
2. **Execution**: Use the `run()` method of the `ScenarioRunner` to start executing your scenario.
3. **Monitoring**: Keep an eye on the logs to understand how your scenario is processing and to catch any issues early.
4. **Review**: After running scenarios, you can review the saved files to analyze the outcomes and make improvements.

## Customizing Scenarios

- **Expand scenarios**: Add more message types or actions to handle new types of interactions.
- **Modify conditions**: Adjust the conditions in `ConditionalScenario` to cater to new logical paths or decisions based on user input or system status.
