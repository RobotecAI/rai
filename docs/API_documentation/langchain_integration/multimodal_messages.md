# Multimodality support

RAI keeps text, images and audio in its own content types, `rai.content.Text`, `rai.content.Image` (a numpy array plus its encoding, e.g. `rgb8`) and `rai.content.Audio` (samples shaped `(n, channels)`, int16 or float32, plus a sample rate). They are converted to LangChain content blocks only at the LLM boundary, in `rai.messages`.

## Content types

::: rai.content.Image

::: rai.content.Audio

## LLM boundary

::: rai.messages.to_blocks

::: rai.messages.from_blocks

::: rai.messages.human

::: rai.messages.with_tool_images

## Usage

```python
from rai import get_llm
from rai.messages import human, preprocess_image

base64_image = preprocess_image('https://raw.githubusercontent.com/RobotecAI/RobotecGPULidar/develop/docs/image/rgl-logo.png')

llm = get_llm("complex")
llm.invoke([human("Describe the image", [base64_image])]).pretty_print()
```

Tools return images as an artifact, `{"images": [...]}` of `Image`s or base64 strings. Most vendors reject images inside tool messages, so RAI agents show them to the model with `with_tool_images`, per model request, without changing the history.

## See Also

-   [Agents](../agents/overview.md): For more information on the different types of agents in RAI
-   [Aggregators](../aggregators/overview.md): For more information on the different types of aggregators in RAI
-   [Connectors](../connectors/overview.md): For more information on the different types of connectors in RAI
-   [Langchain Integration](../langchain_integration/overview.md): For more information on the LangChain integration within RAI
-   [Runners](../runners/overview.md): For more information on the different types of runners in RAI
