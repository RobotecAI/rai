# ROS Packages

RAI comes with multiple configurable ROS2 packages which can be installed alongside the main distribution.

## RAI_grounding_dino

Package enabling use of [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) -- an open-set detection model with ROS2.

Detailed documentation and installation instructions are available in package [README](../src/rai_grounding_dino/README.md)

## RAI_interfaces

Package containing definition of custom messages and services used in RAI. Should be used as a dependancy if any interfaces are required in another package.
To use add `<exec_depend>rai_interfaces</exec_depend>` to target package's `package.xml`.

## RAI_whoami

Package with robot-self identification capabilities. It includes RAI constitution.
