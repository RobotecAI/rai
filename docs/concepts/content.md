# Content and HRI messages

RAI keeps text, images and audio in its own types and converts them to LangChain content blocks only when a model is called. They hold numpy arrays and know nothing about LLM frameworks or ROS.

## Content types

| Type                | Holds                                                                                                                                          |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `rai.content.Text`  | `text`                                                                                                                                         |
| `rai.content.Image` | `data`, an (H, W) or (H, W, C) array, and its `encoding` with the ROS names: `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8`, `mono16`, `32FC1`, ... |
| `rai.content.Audio` | `samples` shaped (n, channels), int16 or float32 in [-1, 1], and `sample_rate`                                                                 |

<!-- test: run -->

```python
import numpy as np

from rai.content import Audio, Image

image = Image(np.zeros((480, 640, 3), np.uint8), "rgb8")
jpeg_b64 = image.encode(max_side=512)        # base64 JPEG for a model; format="png" for PNG
same = Image.decode(jpeg_b64)                # back to an rgb8 Image
gray = image.to("mono8")                     # color conversion

audio = Audio(np.zeros(16000, np.int16), sample_rate=16000)  # 1-D is mono
print(audio.channels, audio.duration)
wav = audio.to_wav()                         # 16-bit PCM WAV bytes
print(Audio.from_wav(wav).as_float32().samples.dtype)
```

An `Image` built from a received ROS message may be a view of the message data: copy the array to change it. Depth images (`mono16`, `32FC1`) are scaled to 8 bits by `encode`.

## HRIMessage

`rai.communication.HRIMessage` is what a human and a robot say to each other: `text`, `images` (a list of `Image`), `audios` (a list of `Audio`), plus a `communication_id` that groups the messages of one utterance or turn, `seq_no` that orders them and `seq_end` that marks the last one.

`HRIConnector` is the protocol of transports that carry them: `async send(target, message)`, `subscribe(source, callback)` and `close()`. RAI implements it for ROS 2 (`ROS2HRIConnector`), and rai_s2s for sound devices.

```python
import numpy as np

from rai.communication import HRIMessage
from rai.communication.ros2 import ROS2HRIConnector
from rai.content import Image

with ROS2HRIConnector() as hri:
    hri.sync.send("/to_human", "Hello")  # a str is sent as text
    snapshot = Image(np.zeros((480, 640, 3), np.uint8), "rgb8")
    hri.sync.send("/to_human", HRIMessage("This is what I see", images=[snapshot]))
    hri.subscribe("/from_human", print)  # callback per message, in order
    last = hri.sync.receive("/from_human", timeout=5.0)
```

`ROS2HRIConnector` publishes `rai_interfaces/msg/HRIMessage` with the `HRI` QoS (reliable, and subscriptions never drop), on the same runtime as `ROS2Connector`. Received images and audio are numpy views of the ROS message.

## The LLM boundary

`rai.messages` converts between RAI content and LangChain messages. Images are JPEG-encoded only here.

<!-- test: run -->

```python
import numpy as np

from rai.content import Image, Text
from rai.messages import from_blocks, human, to_blocks

picture = Image(np.full((64, 64, 3), 255, np.uint8), "rgb8")
message = human("What is in this picture?", images=[picture])  # a LangChain HumanMessage
blocks = to_blocks([Text("caption"), picture])                  # LangChain content blocks
content = from_blocks(message.content)                          # back to [Text, Image]
print([type(c).__name__ for c in content])
```

-   `human(text, images, audios)` builds a `HumanMessage`. Images may be `Image`s or base64 PNG or JPEG strings.
-   Agents accept `HRIMessage`s directly: `agent.submit(hri_message)` and `agent.ainvoke(hri_message)` convert them with `human`.
-   Tools return images as an artifact, `{"images": [...]}` (`rai.messages.MultimodalArtifact`), with a `content_and_artifact` response format. Most providers reject images inside tool messages, so agents show them to the model with `with_tool_images`, per request, without changing the history.
-   `rai.messages.preprocess_image` loads an image from a path, URL, bytes or array as base64 PNG.

See the [API reference](../api/content.md) for every parameter.
