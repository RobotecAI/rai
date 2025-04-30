# RAI introduction

!!! abstract "Development Status"

    RAI is currently undergoing significant development on the development branch, focusing on version 2.0. This major version update will introduce substantial improvements and is not backward compatible with version 1.0. We are targeting an early May 2025 release for RAI 2.0. For the latest stable release, please refer to the [main branch](https://github.com/RobotecAI/rai/tree/main).

RAI is a flexible AI agent framework to develop and deploy Embodied AI features for your robots.

---

![rai-image](./imgs/RAI_simple_diagram_medium.png)

---

<div style="text-align: center;">
    <a href="https://opensource.org/licenses/Apache-2.0">
        <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
    </a>
    <a href="https://github.com/RobotecAI/rai/releases">
        <img src="https://img.shields.io/github/v/release/RobotecAI/rai" alt="GitHub Release">
    </a>
    <a href="https://github.com/robotecai/rai/graphs/contributors">
        <img src="https://img.shields.io/github/contributors/robotecai/rai" alt="Contributors">
    </a>
    <br>
    <img src="https://img.shields.io/badge/Ubuntu-24.04-orange" alt="Ubuntu 24.04">
    <img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python 3.12">
    <img src="https://img.shields.io/badge/ROS2-jazzy-blue" alt="ROS2 jazzy">
    <br>
    <img src="https://img.shields.io/badge/Ubuntu-22.04-orange" alt="Ubuntu 22.04">
    <img src="https://img.shields.io/badge/Python-3.10-blue" alt="Python 3.10">
    <img src="https://img.shields.io/badge/ROS2-humble-blue" alt="ROS2 humble">
    <br>
    <a href="https://discord.gg/3PGHgTaJSB">
        <img src="https://dcbadge.limes.pink/api/server/https://discord.gg/3PGHgTaJSB" alt="Discord">
    </a>
</div>
---

## RAI Framework

The RAI Framework represents a complete, end-to-end solution for developing and deploying sophisticated AI-powered robotic systems. It supports the full lifecycle of embodied AI development, from initial configuration and testing to deployment and continuous improvement.

Our comprehensive suite of integrated packages enables developers to seamlessly transition from concept to production, offering:

- **Complete Development Lifecycle**: From initial agent development to deployment and fine-tuning
- **Modular Architecture**: Choose and combine components based on your specific needs
- **Production-Ready Tools**: Enterprise-grade packages for simulation, testing, and deployment
- **Extensible Platform**: Easy integration with existing robotics infrastructure and custom solutions
- **Advanced Human-Robot Interaction**: Natural language processing, speech recognition, and intuitive interfaces
- **Rich Multimodal Capabilities**: Seamless integration of voice, vision, and sensor data with real-time processing of multiple input/output streams, native handling of diverse data types, and unified multi-sensory perception and action framework.

The framework's components work in perfect harmony to deliver a robust foundation for your robotics projects:

<div style="text-align: center;"><img src="./imgs/rai_packages.png" width="80%" alt="rai-packages"></div>

## Getting Started

Ready to dive into RAI? Start with a [quick-setup guide](install.md).

Here are two ways to begin your journey:

### Option 1: Try Our Demos

Experience RAI in action through our interactive demos. These showcase real-world applications across different robotic platforms:

- 🚜 [Agricultural Robotics](demos/agriculture.md) - See how RAI handles complex decision-making in orchard environments
- 🤖 [Manipulation Tasks](demos/manipulation.md) - Watch RAI control a Franka Panda arm using natural language
- 🚗 [Autonomous Navigation](demos/rosbot_xl.md) - Explore RAI's capabilities with the ROSbot XL platform
- 🐢 [Turtlebot Integration](demos/turtlebot.md) - Learn how RAI works with the popular Turtlebot platform
- 🎤 [Speech Interaction](demos/taxi.md) - Experience RAI's speech-to-speech capabilities in an autonomous taxi scenario

### Option 2: Build Your Own Solution

Follow our comprehensive [walkthrough](developer_guide/walkthrough.md) to:

- Deploy RAI on your robot and enable natural language interactions
- Extend the framework with custom tools and capabilities
- Implement complex, multi-step tasks using RAI's advanced reasoning

## Communication Protocols

RAI provides first-class support for ROS 2 Humble and Jazzy distributions. While ROS 2 serves as our Tier 1 communication protocol, RAI's architecture includes a powerful abstraction layer that:

- Simplifies communication across different networks and protocols
- Enables seamless integration with various communication backends
- Allows for future protocol extensions while maintaining a consistent interface

This design philosophy means that while RAI is fully compatible with ROS 2, most of its features can be utilized independently of the ROS 2 environment. The framework's modular architecture makes it suitable not only for different robotic platforms but also for non-robotic applications, offering flexibility in deployment across various domains.

### Contributing

You are welcome to contribute to RAI! Please see our [Contribution Guide](CONTRIBUTING.md).

!!! tip "Want to know more?"

    ## Community

    ### Embodied AI Community Group

    RAI is one of the main projects in focus of the [Embodied AI Community Group](https://github.com/ros-wg-embodied-ai). If you would like to join the next meeting, look for it in the [ROS Community Calendar](https://calendar.google.com/calendar/u/0/embed?src=c_3fc5c4d6ece9d80d49f136c1dcd54d7f44e1acefdbe87228c92ff268e85e2ea0@group.calendar.google.com&ctz=Etc/UTC).

    ### Publicity

    - A talk about [RAI at ROSCon 2024](https://vimeo.com/1026029511).

    ### RAI Q&A

    Please take a look at [Q&A](https://github.com/RobotecAI/rai/discussions/categories/q-a).
