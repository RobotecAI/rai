# RAI: Human-Robot Interface

Interactions with the robot are crucial to ensure a full understanding of what is happening and what the robot intends to do. RAI provides two ways to achieve this functionality:

- Voice communication using ASR and TTS models (OpenAI Whisper)
- Text communication using Streamlit

The appropriate communication method should be choosen based on the working environment. If the environment is noisy, the text form is recommended. The voice communication is recommended only in quiet spaces.

For more information see:

- Text interface: [link](./human_robot_interface/text_interface.md)
- Voice interface: [link](./human_robot_interface/voice_interface.md)

## General Architecture

![General HRI interface](./imgs/HRI_interface.png)

The general architecture follows the diagram above. Text is captured from the input source, transported to the HMI, processed according to the given tools and robot's rules, and then sent to the output source.

## Voice Interface

![Voice interface](./imgs/HRI_voice_interface.png)

In the voice interface, the input source is a microphone, while the output source is a speaker. The input is processed using the OpenAI [Whisper](https://platform.openai.com/docs/guides/speech-to-text/quickstart) model (cloud-based, paid), while the output can be produced using [OpenTTS](https://github.com/synesthesiam/opentts) (Apache-2.0, depending on the model used) or [ElevenLabs](https://github.com/elevenlabs/elevenlabs-python) (cloud-based, paid).

## Text Interface

![Text interface](./imgs/HRI_text_interface.png)

The text interface is implemented directly in RAI_HMI using Streamlit. The GUI closely follows standard chat-like conversations, with built-in support for tool integration.
