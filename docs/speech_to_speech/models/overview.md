# Models

## Overview

This package provides three primary types of models:

-   **Voice Activity Detection (VAD)**
-   **Wake Word Detection**
-   **Transcription**

These models are designed with simple and consistent interfaces to allow chaining and integration into audio processing pipelines.

## Model Interfaces

### VAD and Wake Word Detection API

All VAD and Wake Word detection models implement a common `detect` interface:

```python
    def detect(
        self, audio_data: NDArray, input_parameters: dict[str, Any]
    ) -> Tuple[bool, dict[str, Any]]:
```

This design supports chaining multiple models together by passing the output dictionary (`input_parameters`) from one model into the next.

### Transcription API

Transcription models implement the `transcribe` method:

```python
    def transcribe(self, data: NDArray[np.int16]) -> str:
```

This method takes raw audio data encoded as 2-byte integers and returns the corresponding text transcription.

## Included Models

### SileroVAD

-   Open source model: [GitHub](https://github.com/snakers4/silero-vad)
-   No additional setup required
-   Returns a confidence value indicating the presence of speech in the audio

### OpenWakeWord

-   Open source project: [GitHub](https://github.com/dscripka/openWakeWord)
-   Supports predefined and custom wake words
-   Returns `True` when the specified wake word is detected in the audio

### OpenAIWhisper

-   Cloud-based transcription model: [Documentation](https://platform.openai.com/docs/guides/speech-to-text)
-   Requires setting the `OPEN_API_KEY` environment variable
-   Offers language and model customization via the API

### LocalWhisper

-   Local deployment of OpenAI Whisper: [GitHub](https://github.com/openai/whisper)
-   Supports GPU acceleration
-   Same configuration interface as OpenAIWhisper

### FasterWhisper

-   Optimized Whisper variant: [GitHub](https://github.com/SYSTRAN/faster-whisper)
-   Designed for high speed and low memory usage
-   Follows the same API as Whisper models

### ElevenLabs

-   Cloud-based TTS model: [Website](https://elevenlabs.io/)
-   Requires the environment variable `ELEVENLABS_API_KEY` with a valid key

### OpenTTS

-   Open source TTS solution: [GitHub](https://github.com/synesthesiam/opentts)
-   Easy setup via Docker:

```bash
 docker run -it -p 5500:5500 synesthesiam/opentts:en --no-espeak
```

-   Provides a TTS server running on port 5500
-   Supports multiple voices and configurations

## Custom Models

### Voice Detection Models

To implement a custom VAD or Wake Word model, inherit from `rai_asr.base.BaseVoiceDetectionModel` and implement the following methods:

```python
class MyDetectionModel(BaseVoiceDetectionModel):
    def detect(self, audio_data: NDArray, input_parameters: dict[str, Any]) -> Tuple[bool, dict[str, Any]]:
        ...

    def reset(self):
        ...
```

### Transcription Models

To implement a custom transcription model, inherit from `rai_asr.base.BaseTranscriptionModel` and implement:

```python
class MyTranscriptionModel(BaseTranscriptionModel):
    def transcribe(self, data: NDArray[np.int16]) -> str:
        ...
```

### TTS Models

To create a custom TTS model, inherit from `rai_tts.models.base.TTSModel` and implement the required interface:

```python
class MyTTSModel(TTSModel):
    def get_speech(self, text: str) -> AudioSegment:
        ...
        return AudioSegment()

    def get_tts_params(self) -> Tuple[int, int]:
        ...
        return sample_rate, channels
```
