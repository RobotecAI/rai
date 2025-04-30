# Speech-to-speech interaction with autonomous taxi

This demo showcases a speech-to-speech interaction with an autonomous taxi using RAI in an AWSIM
environment with Autoware. Users can specify destinations verbally, and the system will process the
request, plan the route, and navigate the taxi accordingly.

!!! note "Work in progress"

    This readme is a work in progress.

## Prerequisites

Before running this demo, ensure you have the following prerequisites installed:

1. Autoware and AWSIM [link](https://tier4.github.io/AWSIM/GettingStarted/QuickStartDemo/) as well
   as you have configured the speech to speech as in
   [speech to speech doc](../human_robot_interface/voice_interface.md)

## Running the Demo

1. Start AWSIM and Autoware:

2. Run the taxi demo:

   ```bash
   source ./setup_shell.sh
   ros2 launch examples/taxi-demo.launch.py
   ```

3. To interact with the taxi using speech, speak your destination into your microphone. The system
   will process your request and plan the route for the autonomous taxi.

## How it works

The taxi demo utilizes several components:

1. Speech recognition (ASR) to convert user's spoken words into text.
2. RAI agent to process the request and interact with Autoware for navigation.
3. Text-to-speech (TTS) to convert the system's response back into speech.
4. Autoware for autonomous driving capabilities.
5. AWSIM for simulation of the urban environment.

The main logic of the demo is implemented in the `TaxiDemo` class, which can be found in:

```python
examples/taxi-demo.py
```
