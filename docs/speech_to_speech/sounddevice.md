# Sound Device Connector

The `SoundDeviceConnector` provides a Human-Robot Interface (HRI) for audio streaming, playback, and recording using sound devices. It is designed for seamless integration with RAI agents and tools requiring audio input/output, and conforms to the generic `HRIConnector` interface.

| Connector              | Description                                                                                  | Example Usage                           |
| ---------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------- |
| `SoundDeviceConnector` | Audio streaming, playback, and recording via sounddevice. Implements HRIConnector for audio. | `connector = SoundDeviceConnector(...)` |

## Key Features

-   Audio playback (write) and recording (read) with flexible device configuration
-   Asynchronous (streaming) and synchronous (service call) audio operations
-   Thread-safe device management and clean shutdown
-   Unified message type (`SoundDeviceMessage`) for audio and control
-   Full support for the HRIConnector interface

## Initialization

To use the connector, specify the target (output) and source (input) devices with their configurations:

```python
from rai_s2s.sound_device import SoundDeviceConfig, SoundDeviceConnector

# Example device configurations
output_config = SoundDeviceConfig(device_name="Speaker", channels=1)
input_config = SoundDeviceConfig(device_name="Microphone", channels=1)

connector = SoundDeviceConnector(
    targets=[("speaker", output_config)],
    sources=[("mic", input_config)],
)
```

> [!WARNING]
> It is not recommended to use device_name set to `default` in `SoundDeviceConfig` due to potential issues with audio.

## Message Type: `SoundDeviceMessage`

```python
from rai_s2s.sound_device import SoundDeviceMessage

msg = SoundDeviceMessage(
    audios=[audio_data],   # List of audio data (bytes or numpy arrays)
    read=False,            # Set True for recording
    stop=False,            # Set True to stop playback/recording
    duration=2.0           # Recording duration (seconds), if applicable
)
```

## Example Usage

### Audio Playback (Synchronous)

```python
# Play audio synchronously
msg = SoundDeviceMessage(audios=[audio_data])
connector.send_message(msg, target="speaker")
```

### Audio Recording (Synchronous)

```python
# Record audio synchronously (blocking)
msg = SoundDeviceMessage(read=True)
recorded_msg = connector.service_call(msg, target="mic", duration=2.0)
recorded_audio = recorded_msg.audios[0]
```

### Audio Streaming (Asynchronous)

```python
# Start asynchronous audio recording
msg = SoundDeviceMessage(read=True)
def on_feedback(audio_chunk):
    print("Received chunk", audio_chunk)
def on_done(final_audio):
    print("Recording finished")
action_handle = connector.start_action(
    action_data=msg,
    target="mic",
    on_feedback=on_feedback,
    on_done=on_done
)

# Stop the stream when done
connector.terminate_action(action_handle)
```

## Device Management

-   Configure devices at initialization or using `configure_device(target, config)`.
-   Retrieve audio parameters using `get_audio_params(target)`.
-   All devices are managed in a thread-safe way and are properly closed on `shutdown()`.

## Error Handling

-   All methods raise `SoundDeviceError` on invalid operations (e.g., unsupported message types, missing audio data).
-   Use `send_message` with `stop=True` to stop playback or recording.
-   `receive_message` is not supported (use actions or service calls for recording).

## See Also

-   [Connectors Overview](../API_documentation/connectors/overview.md)
-   [Agents](../API_documentation/agents/overview.md)
