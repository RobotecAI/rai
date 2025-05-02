# RAI Text To Speech

This is the [RAI](https://github.com/RobotecAI/rai) text to speech package.
It contains Agent definitions for the TTS feature.

## Models

Out of the box the following models are supported:

### ElevenLabs

[ElevenLabs](https://elevenlabs.io/) is a proprietary cloud provider for TTS. Refer to the website for the documentation.
In order to use it the `ELEVENLABS_API_KEY` environment variable must be set, with a valid API key.

### OpenTTS

[OpenTTS](https://github.com/synesthesiam/opentts) is an open source model for TTS.
It can be easily set up using docker. Run:

```
 docker run -it -p 5500:5500 synesthesiam/opentts:en --no-espeak
```

To setup a basic english OpenTTS server on port 5500 (default).
Refer to the providers documentation for available voices and options.

### Custom Models

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

## Agents

### TextToSpeechAgent

The TextToSpeechAgent utilises ROS 2 and sounddevice `Connectors` to receive data, and play it using a speaker.
It complies to the following ROS 2 API:

Subscription topic `/to_human: [rai_interfaces/msg/HRIMessage]`:
`message.text` will be parsed, run through the TTS model and played using the speaker
Subscription topic `/voice_commands: [std_msgs/msg/String]`:
The following values are accepted:

-   `"play"`: allow for playing the voice through the speaker (if voice queue is not empty)
-   `"pause"`: pause the playing of the voice through the speaker
-   `"stop"`: stop the current playback and clear the queue
-   `"tog_play"`: toggle between play and pause
