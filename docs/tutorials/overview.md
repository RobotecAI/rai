# RAI Tutorials Overview

This directory contains a collection of tutorials that guide you through various aspects of the RAI (Robot AI) framework. Each tutorial focuses on different components and use cases of the system.

## Available Tutorials

### 1. [Walkthrough](./walkthrough.md)

A step-by-step guide to creating and deploying a custom RAI agent on a ROS 2 enabled robot. This tutorial covers:

-   Creating a custom RAI Agent from scratch
-   Implementing platform-specific tools for robot control
-   Building an optimized system prompt using rai whoami
-   Deploying and interacting with the agent

### 2. [Create Robot's Whoami](./create_robots_whoami.md)

Learn how to configure RAI to understand your robot's identity, including its appearance, purpose, ethical code, equipment, capabilities, and documentation. This tutorial covers:

-   Setting up the robot's `whoami` package
-   Building embodiment information
-   Testing the configuration with ROS 2 services
-   Using the whoami tools in Python code

### 3. [Tools](./tools.md)

A comprehensive guide to tool development and usage in RAI. This tutorial covers:

-   Understanding the fundamental concepts of tools in LangChain
-   Creating custom tools using both `BaseTool` class and `@tool` decorator
-   Implementing single-modal and multimodal tools
-   Developing ROS 2 specific tools
-   Tool initialization and configuration
-   Using tools in both distributed and local setups

### 4. [Voice Interface](./voice_interface.md)

Learn how to implement human-robot interaction through voice commands. This tutorial covers:

-   Setting up Automatic Speech Recognition (ASR) agent
-   Configuring Text-to-Speech (TTS) agent
-   Running a complete speech-to-speech communication example

## Getting Started

To get started with RAI, we recommend following these tutorials in the following order:

1. Begin with the [Walkthrough](./walkthrough.md) to understand the basic concepts and create your first agent
2. Learn about [Tools](./tools.md) to understand how to extend your agent's capabilities
3. Configure your robot's identity using [Create Robot's Whoami](./create_robots_whoami.md)
4. Add voice interaction capabilities using the [Voice Interface](./voice_interface.md) tutorial

Each tutorial includes practical examples and code snippets to help you implement the concepts in your own projects.
