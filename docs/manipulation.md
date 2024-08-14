# AI-powered robotics manipulation

## Currently planned and developed pipeline

![pipeline](imgs/openvla_diagram.gif)

### **Automated Dataset Generation**

We aim to automatically generate a dataset with various scenes and robot's actions to fine-tune [openVLA](https://openvla.github.io/) - a state-of-the-art Open-Source Vision-Language-Action Model for generalist robot manipulation policies. We aim to leverage our **LLM-powered Scene Generator** enabling easy generation of various scenarios based on user prompts.

**LLM-powered Scene Generator** generates the Python code that creates the prompted scene and makes objects make desired actions. We use the most efficient [Claude models](https://www.anthropic.com/claude) (currently Claude 3.5 Sonnet). The tool is integrated with [O3DE - an open source, real-time 3D engine](https://o3de.org/industries/robotics-and-simulations/).

![manipulation_examples](imgs/manipulation_demo.gif)

### OpenVLA fine-tuning and inference

We integrated openVLA model with [O3DE Engine](https://o3de.org/industries/robotics-and-simulations/) and [ROS2](https://github.com/ros2) to efficiently evaluate the fine-tuning results on simulated robot. Our Robotic Arm Manipulation Demo is available [here](https://github.com/RobotecAI/rai-manipulation-demo). Instructions on how to run the demo will be released soon.

## Issues and challenges

We are intensively working on determining the proper dataset size and diversity to optimally generalize the model for our needs. As openVLA is a very recent state-of-the-art publication, intuitions regarding the fine-tuning of this model are only just being developed.
