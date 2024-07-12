# RAI ASR

## Description

The RAI ASR (Automatic Speech Recognition) node utilizes a combination of voice activity detection (VAD) and a speech recognition model to transcribe spoken language into text. The node is configured to handle multiple languages and model types, providing flexibility in various ASR applications. It detects speech, records it, and then uses a model to transcribe the recorded audio into text.

## Installation

```bash
rosdep install --from-paths src --ignore-src -r
```

## Subscribed Topics

This node does not subscribe to any topics. It operates independently, capturing audio directly from the microphone.

## Published Topics

- **`transcription`** (`std_msgs/String`): Publishes the transcribed text obtained from the audio recording.

## Parameters

- **`language`** (`string`, default: `"en"`): The language code for the ASR model. This parameter defines the language in which the audio will be transcribed.
- **`model`** (`string`, default: `"base"`): The type of ASR model to use. Different models may have different performance characteristics. For list of models see `python -c "import whisper;print(whisper.available_models())"`
- **`silence_grace_period`** (`double`, default: `1.0`): The grace period in seconds after silence is detected to stop recording. This helps in determining the end of a speech segment.
- **`sample_rate`** (`integer`, default: `0`): The sample rate for audio capture. If set to 0, the sample rate will be auto-detected.
