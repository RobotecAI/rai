# RAI Speech To Speech

## RAI ASR

### Description

This is the [RAI](https://github.com/RobotecAI/rai) automatic speech recognition package.
It contains Agents definitions for the ASR feature.

### Models

This package contains three types of models: Voice Activity Detection (VAD), Wake word and transcription.

The `detect` API for VAD and Wake word models, with the following signature:

```
    def detect(
        self, audio_data: NDArray, input_parameters: dict[str, Any]
    ) -> Tuple[bool, dict[str, Any]]:
```

Allows for chaining the models into detection piplelines. The `input_parameters` provide a utility to pass the output dictionary from previous models.

The `transcribe` API for transcription models, with the following signature:

```
    def transcribe(self, data: NDArray[np.int16]) -> str:
```

Takes the audio data encoded as 2 byte ints and returns the string with transcription.

#### SileroVAD

[SileroVAD](https://github.com/snakers4/silero-vad) is an open source VAD model. It requires no additional setup. It returns confidence regarding there being voice in the provided recording.

#### OpenWakeWord

[OpenWakeWord](https://github.com/dscripka/openWakeWord) is an open source package containing multiple pre-configured models, as well as allowing for using custom wake words.
Refer to the package documentation for adding custom wake words.

The model is expected to return `True` if the wake word is detected in the audio sample contains it.

#### OpenAIWhisper

[OpenAIWhisper](https://platform.openai.com/docs/guides/speech-to-text) is a cloud-based transcription model. Refer to the documentation for configuration capabilities.
The environment variable `OPEN_API_KEY` needs to be set to a valid OPENAI key in order to use this model.

#### LocalWhisper

[LocalWhisper](https://github.com/openai/whisper) is the locally hosted version of OpenAI whisper. It supports GPU acceleration, and follows the same configuration capabilities, as the cloud based one.

#### FasterWhisper

[FasterWhisper](https://github.com/SYSTRAN/faster-whisper) is another implementation of the whisper model. It's optimized for speed and memory footprint. It follows the same API as the other two provided implementations.

#### Custom Models

Custom VAD, Wake Word, or other detection models can be implemented by inheriting from `rai_asr.base.BaseVoiceDetectionModel`. The `detect` and `reset` methods must be implemented.

Custom transcription models can be implemented by inheriting from `rai_asr.base.BaseTranscriptionModel`. The `transcribe` method must be implemented.

### Agents

#### Speech Recognition Agent

The speech recognition Agent uses ROS 2 and sounddevice `Connectors`, to communicate with other agents and access the microphone.

It fulfills the following ROS 2 communication API:

Publishes to topic `/to_human: [HRIMessage]`:
`message.text` is set with the transcription result using the selected transcription model.

Publishes to topic `/voice_commands: [std_msgs/msg/String]`:

- `"pause"` - when voice is detected but the `detection_pipeline` didn't return detection (for interruptive S2S)
- `"play"` - when voice is not detected, but there was previously a transcription sent
- `"stop"` - when voice is detected and the `detection_pipeline` returned a detection (or is empty)

## RAI Text To Speech

This is the [RAI](https://github.com/RobotecAI/rai) text to speech package.
It contains Agent definitions for the TTS feature.

### Models

Out of the box the following models are supported:

#### ElevenLabs

[ElevenLabs](https://elevenlabs.io/) is a proprietary cloud provider for TTS. Refer to the website for the documentation.
In order to use it the `ELEVENLABS_API_KEY` environment variable must be set, with a valid API key.

#### OpenTTS

[OpenTTS](https://github.com/synesthesiam/opentts) is an open source model for TTS.
It can be easily set up using docker. Run:

```
 docker run -it -p 5500:5500 synesthesiam/opentts:en --no-espeak
```

To setup a basic english OpenTTS server on port 5500 (default).
Refer to the providers documentation for available voices and options.

#### Custom Models

To add your custom TTS model inherit from the `rai_tts.models.base.TTSModel` class.

You can use the following template:

```
class MyTTSModel(TTSModel):
    def get_speech(self, text: str) -> AudioSegment:
        ...
        return AudioSegment()

    def get_tts_params(self) -> Tuple[int, int]:
        ...
        return sample_rate, channels

```

Such a model will work with the `TextToSpeechAgent` defined below:

### Agents

#### TextToSpeechAgent

The TextToSpeechAgent utilises ROS 2 and sounddevice `Connectors` to receive data, and play it using a speaker.
It complies to the following ROS 2 API:

Subscription topic `/to_human: [rai_interfaces/msg/HRIMessage]`:
`message.text` will be parsed, run through the TTS model and played using the speaker
Subscription topic `/voice_commands: [std_msgs/msg/String]`:
The following values are accepted:

- `"play"`: allow for playing the voice through the speaker (if voice queue is not empty)
- `"pause"`: pause the playing of the voice through the speaker
- `"stop"`: stop the current playback and clear the queue
- `"tog_play"`: toggle between play and pause
