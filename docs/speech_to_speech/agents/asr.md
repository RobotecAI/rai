# SpeechRecognitionAgent

## Overview

The `SpeechRecognitionAgent` in the RAI framework is a specialized agent that performs voice activity detection (VAD), audio recording, and transcription. It integrates tightly with audio input sources and ROS2 messaging, allowing it to serve as a real-time voice interface for robotic systems.

This agent manages multiple pipelines for detecting when to start and stop recording, performs transcription using configurable models, and broadcasts messages to relevant ROS2 topics.

## Class Definition

??? info "SpeechRecognitionAgent class definition"

    ::: rai_s2s.asr.agents.asr_agent.SpeechRecognitionAgent

## Purpose

The `SpeechRecognitionAgent` class enables real-time voice processing with the following responsibilities:

-   Detecting speech through VAD
-   Managing recording state and grace periods
-   Buffering and threading transcription processes
-   Publishing transcriptions and control messages to ROS2 topics
-   Supporting multiple VAD and transcription model types

## Initialization Parameters

| Parameter             | Type                       | Description                                                                   |
| --------------------- | -------------------------- | ----------------------------------------------------------------------------- |
| `microphone_config`   | `SoundDeviceConfig`        | Configuration for the microphone input.                                       |
| `ros2_name`           | `str`                      | Name of the ROS2 node.                                                        |
| `transcription_model` | `BaseTranscriptionModel`   | Model instance for transcribing speech.                                       |
| `vad`                 | `BaseVoiceDetectionModel`  | Model for detecting voice activity.                                           |
| `grace_period`        | `float`                    | Time (in seconds) to continue buffering after speech ends. Defaults to `1.0`. |
| `logger`              | `Optional[logging.Logger]` | Logger instance. If `None`, defaults to module logger.                        |

## Key Methods

### `from_config()`

Creates a `SpeechRecognitionAgent` instance from a YAML config file. Dynamically loads the required transcription and VAD models.

### `run()`

Starts the microphone stream and handles incoming audio samples.

### `stop()`

Stops the agent gracefully, joins all running transcription threads, and shuts down ROS2 connectors.

### `add_detection_model(model, pipeline="record")`

Adds a custom VAD model to a processing pipeline.

-   `pipeline` can be either `'record'` or `'stop'`

!!! note "`'stop'` pipeline"

    The `'stop'` pipeline is present for forward compatibility. It currently doesn't affect Agent's functioning.

## Best Practices

1. **Graceful Shutdown**: Always call `stop()` to ensure transcription threads complete.
2. **Model Compatibility**: Ensure all transcription and VAD models are compatible with the sample rate (typically 16 kHz).
3. **Thread Safety**: Use provided locks for shared state, especially around the transcription model.
4. **Logging**: Utilize `self.logger` for debug and info logs to aid in tracing activity.
5. **Config-driven Design**: Use `from_config()` to ensure modular and portable deployment.

## Architecture

The `SpeechRecognitionAgent` typically interacts with the following components:

-   **SoundDeviceConnector**: Interfaces with microphone audio input.
-   **BaseVoiceDetectionModel**: Determines whether speech is present.
-   **BaseTranscriptionModel**: Converts speech audio into text.
-   **ROS2Connector / ROS2HRIConnector**: Publishes transcription and control messages to ROS2 topics.
-   **Config Loader**: Dynamically creates agent from structured config files.

## See Also

-   [BaseAgent](../agents/overview.md): Abstract agent class providing lifecycle and logging support.
-   [ROS2 Connectors](../connectors/ros2_connector.md): Communication layer for ROS2 topics.
-   [Models](../models/overview.md): For available voice based models and instructions for creating new ones.
-   [TextToSpeech](tts.md): For TextToSpeechAgent meant for distributed deployment.
