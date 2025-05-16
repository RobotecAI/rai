# S2S Agents

## Overview

Agents in RAI are modular components that encapsulate specific functionalities and behaviors. They follow a consistent interface defined by the `BaseAgent` class and can be combined to create complex robotic systems. The Speech to Speech Agents are used for voice-based interaction, and communicate with other agents.

## SpeechToSpeechAgent

`SpeechToSpeechAgent` is the abstract base class for locally deployable S2S Agents. It provides functionality to manage sound device integration, as well as defines the communication schema for integration with the rest of the system.

### Class Definition

??? info "SpeechToSpeechAgent class definition"

    ::: rai_s2s.s2s.agents.s2s_agent.SpeechToSpeechAgent

### Communication

The Agent communicates through two communication channels provided during initialization - `from_human` and `to_human`.
On the `from_human` channel text transcribed from human voice is published.
On the `to_human` channel receives text to be played to the human through text-to-speech.

### Voice interaction

The voice interaction is performed through two audio streams, with two devices.
These devices can be different, but don't have to - and in case of most local deployments they will be the same.
The list of available sounddevices for configuration can be obtained by running `python -c "import sounddevice as sd; print(sd.query_devices())"`.
The configuration requires the user to specify the name of the sound device to be used for interfacing.
This is the entire string from the index until the comma before the hostapi (typically `ALSA` on Ubuntu).

The voice interaction works as follows: - The user speaks, which leads to the `VoiceActivityDetection` model activation. - \[Optional\] the recording pipeline (containing other models like [OpenWakeWord](models/overview.md)) runs checks. - The recording starts. - The recording continues until the user stops talking (based on silence grace period). - The recording is transcribed and sent to the system. - The Agent receives text data to be played to the user. - The playback begins. - The playback can be interrupted by user speaking: - if there is additional recording pipeline the playback will pause while the user speaks (and continue, if the pipeline returns false). - otherwise the new recording will be send to the system, and transcription will stop the playback.

### Implementations

ROS based implementation is available in `ROS2S2SAgent`.

??? info "ROS2S2SAgent class definition"

    ::: rai_s2s.s2s.agents.ros2s2s_agent.ROS2S2SAgent

## See Also

-   [Models](../models/overview.md): For available voice based models and instructions for creating new ones.
-   [AutomaticSpeechRecognition](asr.md): For AutomaticSpeechRecognitionAgent meant for distributed deployment.
-   [TextToSpeech](tts.md): For TextToSpeechAgent meant for distributed deployment.
