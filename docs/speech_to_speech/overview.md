# Speech To Speech

## Introduction

`rai s2s` provides tools and components for voice interaction with the system. This package contains plug-and-play Agents which can be easily integrated with Agents provided by `rai core`, as well as custom ones. It also provides integration with host sound system, which can be used for low level sound manipulation.

## Core Components

| Component                    | Description                                                                                               |
| ---------------------------- | --------------------------------------------------------------------------------------------------------- |
| [Agents](agents/overview.md) | Agents in `rai s2s` provide functionality for voice interaction with the rest of the system.              |
| [Models](models/overview.md) | `rai s2s` provides a models which can be optionally installed and utilized by the Agents.                 |
| [Connector](sounddevice.md)  | The `sounddevice` connector allows for interfacing directly with sound devices for asynchronous sound IO. |

## Best Practices

When utilizing S2S features:

1. Deployment of `SpeechToSpeechAgent` is meant for local setup, while the `SpeechRecognition` and `TextToSpeech` Agents are meant to be ran on separate hosts.
2. Note that `sounddevice` python API has notable issues in multi-threaded environment - this can lead to issues when developing Agents using the `SoundDeviceConnector`
