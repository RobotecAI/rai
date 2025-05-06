# Multimodality support

RAI implements `MultimodalMessage` that allows using image and audio\* information in langchain.

!!! important "Audio is not fully supported yet"

    Audio is currently added as a placeholder for future implementation.

## Class Definition

LangChain supports multimodal data by default. This is done by expanding the content section from string to dictionary, containing specific keys.
To make it easier to use, RAI implements a `MultimodalMessage` class, which is a wrapper around the `BaseMessage` class.

### Class Definition

::: rai.messages.multimodal.MultimodalMessage

#### Subclasses

::: rai.messages.multimodal.HumanMultimodalMessage

::: rai.messages.multimodal.AIMultimodalMessage

::: rai.messages.multimodal.SystemMultimodalMessage

::: rai.messages.multimodal.ToolMultimodalMessage

## Usage

Example:

```python
from rai.messages import HumanMultimodalMessage, preprocess_image
from rai import get_llm_model # initialize your model of choice defined in config.toml

base64_image = preprocess_image('https://raw.githubusercontent.com/RobotecAI/RobotecGPULidar/develop/docs/image/rgl-logo.png')

llm = get_llm_model(model_type='complex_model') # initialize your vendor of choice in config.toml
msg = [HumanMultimodalMessage(content='Describe the image', images=[base64_image])]
llm.invoke(msg).pretty_print()

# ================================== Ai Message ==================================
#
# The image features the words "Robotec," "GPU," and "Lidar" displayed in a stylized,
# multicolored font against a black background. The text has a wavy, striped pattern,
# incorporating red, green, and blue colors that give it a vibrantly layered appearance.
```

Implementation of the following messages is identical: HumanMultimodalMessage,
SystemMultimodalMessage, AIMultimodalMessage.

!!! warning "ToolMultimodalMessage usage"

    Most of the vendors, do not support multimodal tool messages.
    `ToolMultimodalMessage` has an addition of `postprocess` method, which converts the
    `ToolMultimodalMessage` into format that is compatible with a chosen vendor.

## See Also

-   [Agents](../agents/overview.md): For more information on the different types of agents in RAI
-   [Aggregators](../aggregators/overview.md): For more information on the different types of aggregators in RAI
-   [Connectors](../connectors/overview.md): For more information on the different types of connectors in RAI
-   [Langchain Integration](../langchain_integration/overview.md): For more information on the LangChain integration within RAI
-   [Runners](../runners/overview.md): For more information on the different types of runners in RAI
