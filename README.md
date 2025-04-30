# RAI

RAI is a flexible AI agent framework to develop and deploy Embodied AI features for your robots.

📚 For the latest documentation and tutorials including setup guide, visit [robotecai.github.io/rai](https://robotecai.github.io/rai/)

---

<div align="center">

![rai-image](./docs/imgs/RAI_simple_diagram_medium.png)

---

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub Release](https://img.shields.io/github/v/release/RobotecAI/rai)
![Contributors](https://img.shields.io/github/contributors/robotecai/rai)

![Static Badge](https://img.shields.io/badge/Ubuntu-24.04-orange)
![Static Badge](https://img.shields.io/badge/Ubuntu-22.04-orange)
![Static Badge](https://img.shields.io/badge/Python-3.12-blue)
![Static Badge](https://img.shields.io/badge/Python-3.10-blue)
![Static Badge](https://img.shields.io/badge/ROS2-jazzy-blue)
![Static Badge](https://img.shields.io/badge/ROS2-humble-blue)

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/3PGHgTaJSB)](https://discord.gg/3PGHgTaJSB)

</div>

---

## 🎯 Overview

The RAI framework is designed to revolutionize robotics by:

🤖 **Empowering Multi-Agent Systems**

- Seamlessly integrate Gen AI capabilities into your robots
- Enable sophisticated agent-based architectures

🔄 **Enhancing Robot Intelligence**

- Add natural human-robot interaction capabilities
- Bring flexible problem-solving to your existing stack
- Provide ready-to-use AI features out of the box

🌟 **Supporting Multi-Modal Interaction**

- Handle diverse data types natively
- Enable rich sensory integration
- Process multiple input/output modalities simultaneously

## RAI framework

- [x] rai core: Core functionality for multi-agent system, human-robot interaction and
      multi-modalities.
- [x] rai whoami: Tool to extract and synthesize robot embodiment information from a structured
      directory of documentation, images, and URDFs.
- [x] rai_asr: Speech-to-text models and tools.
- [x] rai_tts: Text-to-speech models and tools.
- [x] rai_sim: Package for connecting RAI to simulation environments.
- [x] rai_bench: Benchmarking suite for RAI. Test agents, models, tools, simulators, etc.
- [x] rai_openset: Openset detection models and tools.
- [x] rai_nomad: Integration with NoMaD for navigation.
- [ ] rai_finetune: Finetune LLMs on your embodied data.

### Simulation demos

| Try RAI yourself with these demos: | Application | Robot | Description | Docs Link |     |
| ---------------------------------- | ----------- | ----- | ----------- | --------- | --- |

---

| ------------------------------------------------------------- | | Mission and obstacle reasoning
in orchards | Autonomous tractor | In a beautiful scene of a virtual orchard, RAI goes beyond
obstacle detection to analyze best course of action for a given unexpected situation. |
[link](docs/demos/agriculture.md) | | Manipulation tasks with natural language | Robot Arm (Franka
Panda) | Complete flexible manipulation tasks thanks to RAI and Grounded SAM 2 |
[link](docs/demos/manipulation.md) | | Autonomous mobile robot demo | Husarion ROSbot XL |
Demonstrate RAI's interaction with an autonomous mobile robot platform for navigation and control |
[link](docs/demos/rosbot_xl.md) | | Turtlebot demo | Turtlebot | Showcase RAI's capabilities with
the popular Turtlebot platform | [link](docs/demos/turtlebot.md) | | Speech-to-speech interaction
with autonomous taxi | Simulated car | Demonstrate RAI's speech-to-speech interaction capabilities
for specifying destinations to an autonomous taxi in awsim with autoware environment |
[link](docs/demos/taxi.md) |

## Community

### Embodied AI Community Group

RAI is one of the main projects in focus of the
[Embodied AI Community Group](https://github.com/ros-wg-embodied-ai). If you would like to join the
next meeting, look for it in the
[ROS Community Calendar](https://calendar.google.com/calendar/u/0/embed?src=c_3fc5c4d6ece9d80d49f136c1dcd54d7f44e1acefdbe87228c92ff268e85e2ea0@group.calendar.google.com&ctz=Etc/UTC).

### Publicity

- A talk about [RAI at ROSCon 2024](https://vimeo.com/1026029511).

### RAI Q&A

Please take a look at [Q&A](https://github.com/RobotecAI/rai/discussions/categories/q-a).

### Developer Resources

See our [Developer Guide](docs/developer_guide.md) for a deeper dive into RAI, including
instructions on creating a configuration specifically for your robot.

### Contributing

You are welcome to contribute to RAI! Please see our [Contribution Guide](CONTRIBUTING.md).
