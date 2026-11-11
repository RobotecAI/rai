# Human Robot Interface via Voice

RAI provides two ROS enabled agents for Speech to Speech communication.

## Automatic Speech Recognition Agent

See `examples/s2s/asr.py` for an example usage.

The agent requires configuration of `sounddevice` and `ros2` connectors as well as a required voice
activity detection (eg. `SileroVAD`) and transcription model e.g. (`LocalWhisper`), as well as
optionally additional models to decide if the transcription should start (e.g. `OpenWakeWord`).

The Agent publishes information on two topics:

`/from_human`: `rai_interfaces/msg/HRIMessages` - containing transcriptions of the recorded speech

`/voice_commands`: `std_msgs/msg/String` - containing control commands, to inform the consumer if
speech is currently detected (`{"data": "pause"}`), was detected, and now it stopped
(`{"data": "play"}`), and if speech was transcribed (`{"data": "stop"}`).

The Agent utilises sounddevice module to access user's microphone, by default the `"default"` sound
device is used. To get information about available sounddevices use:

```
python -c "import sounddevice; print(sounddevice.query_devices())"
```

The device can be identifed by name and passed to the configuration.

## TextToSpeechAgent

See `examples/s2s/tts.py` for an example usage.

The agent requires configuration of `sounddevice` and `ros2` connectors as well as a required
TextToSpeech model (e.g. `OpenTTS`). The Agent listens for information on two topics:

`/to_human`: `rai_interfaces/msg/HRIMessages` - containing responses to be played to human. These
responses are then transcribed and put into the playback queue.

`/voice_commands`: `std_msgs/msg/String` - containing control commands, to pause current playback
(`{"data": "pause"}`), start/continue playback (`{"data": "play"}`), or stop the playback and drop
the current playback queue (`{"data": "play"}`).

The Agent utilises sounddevice module to access user's speaker, by default the `"default"` sound
device is used. To get a list of names of available sound devices use:

```
python -c 'import sounddevice as sd; print([x["name"] for x in list(sd.query_devices())])'
```

The device can be identifed by name and passed to the configuration.

### OpenTTS

To run OpenTTS (and the example) a docker server containing the model must be running.

To start it run:

```
docker run -it -p 5500:5500 synesthesiam/opentts:en --no-espeak
```

## Running example

To run the provided example of S2S configuration with a minimal LLM-based agent run in 4 separate
terminals:

```
$ docker run -it -p 5500:5500 synesthesiam/opentts:en --no-espeak
$ python ./examples/s2s/asr.py
$ python ./examples/s2s/tts.py
$ python ./examples/s2s/conversational.py
```
