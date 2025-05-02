# Multimodality support

RAI implements `MultimodalMessage` that allows using image and audio\* information in langchain.

!!! important "Audio is not fully supported yet"

    Audio is currently added as a placeholder for future implementation.

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
