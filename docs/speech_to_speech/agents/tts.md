# TextToSpeechAgent

## Overview

The `TextToSpeechAgent` in the RAI framework is a modular agent responsible for converting incoming text into audio using a text-to-speech (TTS) model and playing it through a configured audio output device. It supports real-time playback control through ROS2 messages and handles asynchronous speech processing using threads and queues.

## Class Definition

??? info "TextToSpeechAgent class definition"

    ::: rai_s2s.tts.agents.TextToSpeechAgent

## Purpose

The `TextToSpeechAgent` enables:

-   Real-time conversion of text to speech
-   Playback control (play/pause/stop) via ROS2 messages
-   Dynamic loading of TTS models from configuration
-   Robust audio handling using queues and event-driven logic
-   Integration with human-robot interaction topics (HRI)

## Initialization Parameters

| Parameter            | Type                       | Description                                             |
| -------------------- | -------------------------- | ------------------------------------------------------- |
| `speaker_config`     | `SoundDeviceConfig`        | Configuration for the audio output (speaker).           |
| `ros2_name`          | `str`                      | Name of the ROS2 node.                                  |
| `tts`                | `TTSModel`                 | Text-to-speech model instance.                          |
| `logger`             | `Optional[logging.Logger]` | Logger instance, or default logger if `None`.           |
| `max_speech_history` | `int`                      | Number of speech message IDs to remember (default: 64). |

## Key Methods

### `from_config(cfg_path: Optional[str])`

Instantiates the agent from a configuration file, dynamically selecting the TTS model and setting up audio output.

### `run()`

Initializes the agent:

-   Starts a thread to handle queued text-to-speech conversion
-   Launches speaker playback via `SoundDeviceConnector`

### `stop()`

Gracefully stops the agent by setting the termination flag and joining the transcription thread.

## Communication

The Agent uses the `ROS2HRIConnector` for connection through 2 ROS2 topics:

-   `/to_human`: Incoming text messages to convert. Uses `rai_interfaces/msg/HRIMessage`.
-   `/voice_commands`: Playback control with ROS2 `std_msgs/msg/String`. Valid values: `"play"`, `"pause"`, `"stop"`

## Best Practices

1. **Queue Management**: Properly track transcription IDs to avoid queue collisions or memory leaks.
2. **Playback Sync**: Ensure audio queues are flushed on `stop` to avoid replaying outdated speech.
3. **Graceful Shutdown**: Always call `stop()` to terminate threads cleanly.
4. **Model Configuration**: Ensure model-specific settings (e.g., voice selection for ElevenLabs) are defined in config files.

## Architecture

The `TextToSpeechAgent` interacts with the following core components:

-   **TTSModel**: Converts text into audio (e.g., ElevenLabsTTS, OpenTTS)
-   **SoundDeviceConnector**: Sends synthesized audio to output hardware
-   **ROS2HRIConnector**: Handles incoming HRI and command messages
-   **Queues and Threads**: Enable asynchronous and buffered audio processing

## See Also

-   [BaseAgent](../agents/overview.md#baseagent): Abstract base for all agents in RAI
-   [SoundDeviceConnector](../connectors/sound_device_connector.md): For details on speaker configuration and streaming
-   [Text-to-Speech Models](../models/tts_models.md): Supported TTS engines and usage
-   [ROS2 HRI Messaging](../connectors/ros2_connector.md): Interfacing with `/to_human` and `/voice_commands`
-   [Agent Configuration](../configuration/overview.md): Configuring TTS agents using YAML
