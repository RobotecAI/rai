# 📘 README for AI Vendor Integration Module

## Overview

This module provides a framework for integrating with different AI vendors. It defines a base class for AI vendor interactions and specific implementations for various vendors such as OpenAI, Ollama, and AWS Bedrock. Each vendor class is responsible for managing API calls, handling messages, and logging interactions.

## Classes

### `AiVendor`

#### Overview

`AiVendor` is an abstract base class that defines the structure for AI vendor interactions. It initializes common parameters and provides an abstract method `call_api` that must be implemented by subclasses.

#### Parameters

- `model` (str): The model identifier to be used for API calls.
- `stream` (bool): Indicates whether streaming is enabled.
- `logging_level` (int): Sets the logging level for the class.

#### Methods

- `call_api(messages: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]`: Abstract method to call the AI vendor's API.

### `OllamaVendor`

#### Overview

`OllamaVendor` extends `AiVendor` to provide specific implementations for interacting with the Ollama API. This class handles message formatting and API call retries in case of connection errors.

#### Parameters

- `ip_address` (str): The IP address of the Ollama service.
- `port` (str): The port on which the Ollama service is running.
- Inherits parameters from `AiVendor`.

#### Methods

- `call_api(messages: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]`: Implements the API call to the Ollama service, with retry logic for connection errors.

### `OpenAIVendor`

#### Overview

`OpenAIVendor` extends `AiVendor` to interact with the OpenAI API. It manages API key initialization, message building, and cost calculation based on token usage.

#### Parameters

- Inherits parameters from `AiVendor`.

#### Methods

- `call_api(messages: List[Dict[str, Any]], max_tokens: int = 1000) -> Dict[str, Any]`: Implements the API call to OpenAI, including message formatting and cost calculation.
- `_build_message(message: Dict[str, Any]) -> Dict[str, Any]`: Formats messages for the OpenAI API.
- `_build_text_message(user: str, text: str) -> Dict[str, Any]`: Builds a text message.
- `_build_image_message(user: str, text: str, images: List[str]) -> Dict[str, Any]`: Builds a message with images.

### `AWSBedrockVendor`

#### Overview

`AWSBedrockVendor` extends `AiVendor` to interact with the AWS Bedrock service. This class handles session management, message preprocessing, and API calls.

#### Parameters

- `system_prompt_allowed` (bool): Indicates if system prompts are allowed.
- `squash_messages` (bool): Indicates if messages should be squashed into a single message.
- Inherits parameters from `AiVendor`.

#### Methods

- `call_api(messages: List[Dict[str, Any]], max_tokens: int = 1000) -> Dict[str, Any]`: Implements the API call to AWS Bedrock, including message preprocessing.
- `_preprocess_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]`: Preprocesses messages by optionally squashing them and handling system prompts.
- `_squash_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]`: Squashes multiple messages into a single message.
- `_build_message(message: Dict[str, Any]) -> Dict[str, Any]`: Formats messages for the AWS Bedrock API.
- `_build_text_message(user: str, text: str) -> Dict[str, Any]`: Builds a text message.
- `_build_image_message(user: str, text: str, images: List[str]) -> Dict[str, Any]`: Builds a message with images.

## Logging

Each class utilizes the Python `logging` module to log interactions with the respective AI vendor. Logging levels and messages can be adjusted as needed for debugging and monitoring purposes.

TODO:

- [ ] refactor logging cost

## Usage

To use any of the vendor classes, instantiate the class with the required parameters and call the `call_api` method with the appropriate messages and token limits.

Example:

```python
openai_vendor = OpenAIVendor(model="gpt-3.5-turbo", stream=False, logging_level=logging.INFO)
response = openai_vendor.call_api(messages=[{"role": "user", "content": "Hello"}], max_tokens=100)
print(response)
```
