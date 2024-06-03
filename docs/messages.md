# Message Types

## ConstantMessage

`ConstantMessage` represents a message with fixed content. It requires the `role` (e.g., "user") and `content` parameters. An optional `images` parameter can be used to attach images to the message. Images can be either a path to a locally stored image or a URL pointing to a website location. The link should start with either `http` or `https`.

```python
ConstantMessage(
    role="user",
    content="Your content here",
    images=[Message.preprocess_image("path/to/image.jpg")],
)
```

## UserMessage

`UserMessage` is a specific type of `ConstantMessage` that represents a message from the user. It simplifies creating user messages by setting the `role` to "user" by default.

```python
UserMessage(
    content="Your content here",
    images=[Message.preprocess_image("path/to/image.jpg")], # optional
)
```

## SystemMessage

`SystemMessage` is a specific type of `ConstantMessage` that represents a message from the system. It simplifies creating system messages by setting the `role` to "system" by default.

```python
SystemMessage(
    content="System initialization message",
)
```

## AssistantMessage

`AssistantMessage` represents a message from the AI assistant, including requirements that dictate conditions the assistant's response must meet. If these requirements are not met, the system attempts to gather a proper response multiple times. When the severity is set to OPTIONAL, the scenario continues with a warning. For MANDATORY severity, an error is raised.

```python
AssistantMessage(
    requirements=[
        MessageLengthRequirement(severity=RequirementSeverity.OPTIONAL, max_length=5)
    ]
)
```

## ConditionalMessage

`ConditionalMessage` represents a message that depends on a condition. It requires a `condition` function that takes the assistant's previous response as input and returns a boolean. Depending on the result, either the `if_true` or `if_false` message will be used.

```python
ConditionalMessage(
    condition=lambda x: "yes" in x.lower(),
    if_true=Message(
        role="user",
        content="Content if condition is true",
    ),
    if_false=Message(
        role="user",
        content="Content if condition is false",
    ),
)
```
