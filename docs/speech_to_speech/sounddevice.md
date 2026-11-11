# Sound Device Connector

The `SoundDeviceConnector` provides a Human-Robot Interface (HRI) for audio streaming, playback, and recording using sound devices. It is designed for RAI agents and tools requiring audio input/output; audio is `rai.content.Audio`.

| Connector              | Description                                               | Example Usage                           |
| ---------------------- | --------------------------------------------------------- | --------------------------------------- |
| `SoundDeviceConnector` | Audio streaming, playback, and recording via sounddevice. | `connector = SoundDeviceConnector(...)` |

## Key Features

-   Audio playback (write) and recording (read) with flexible device configuration
-   Streaming and single-clip audio operations
-   Audio as `rai.content.Audio` (samples shaped (n, channels) and a sample rate)

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

> [!TIP]
> If you're experiencing audio issues and device_name is set to 'default', try specifying the exact device name instead, as this often resolves the problem.

## Example Usage

### Playback and Recording

```python
connector.play("speaker", audio)                    # returns at once
connector.play("speaker", audio, blocking=True)     # returns when the clip ends
connector.stop("speaker")                           # stops the current clip
recorded = connector.record("mic", duration=2.0)    # blocks, returns Audio
```

### Streaming

```python
def on_samples(samples, status_flags):
    print("Received", len(samples), "samples")

connector.open_input_stream("mic", on_samples)

def fill(outdata, frames, time, status_flags):
    outdata[:] = 0  # write the next `frames` frames here

connector.open_output_stream("speaker", fill)

connector.close_streams()  # every device, or close_streams("mic")
```

## Device Management

-   Configure devices at initialization or using `configure_device(name, config)`.
-   Retrieve audio parameters using `get_audio_params(name)`.
-   `close()` closes the streams of every device.
-   Invalid operations (for example recording on an output-only device) raise `SoundDeviceError`.

## See Also

-   [Connectors Overview](../concepts/connectors.md)
-   [Agents](../concepts/agents.md)
