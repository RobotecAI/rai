# RAI: Human-Robot Interaction

RAI provides a Human-Robot Interaction (HRI) package that enables communication with your robots. This package allows you to chat with your robot, give it tasks, and receive feedback and reports. You have the following options for interaction:

- [Voice communication](human_robot_interface/voice_interface.md) using Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) models
- [Text communication](human_robot_interface/text_interface.md) using [Streamlit](https://streamlit.io)

Voice communication might be challenging in noisy environments. In such cases, it's recommended to use the text channel.

## How it works?

### General Architecture

![General HRI interface](./imgs/HRI_interface.png)

The general architecture follows the diagram above. Text is captured from the input source, transported to the Human-Machine Interface (HMI), processed according to the given tools and robot's rules, and then sent to the output source.

### Voice Interface

![Voice interface](./imgs/HRI_voice_interface.png)

In the voice interface, the input source is a microphone, while the output source is a speaker. The input is processed using the OpenAI [Whisper](https://platform.openai.com/docs/guides/speech-to-text/quickstart) model (cloud-based, paid) or with the local model, while the output can be produced using [OpenTTS](https://github.com/synesthesiam/opentts) (Apache-2.0, depending on the model used) or [ElevenLabs](https://github.com/elevenlabs/elevenlabs-python) (cloud-based, paid).

### Text Interface

![Text interface](./imgs/HRI_text_interface.png)

The text interface is implemented directly in RAI_HMI using Streamlit. The GUI closely follows standard chat-like conversations, with built-in support for tool integration.
