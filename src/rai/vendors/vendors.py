import copy
import json
import logging
import os
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, TypedDict

import boto3
import requests
from openai import OpenAI
from openai.types import CompletionUsage

from rai.message import Message


class Response(TypedDict):
    role: str
    content: List[Dict[str, Any]]


class AiVendor:
    def __init__(self, model: str, stream: bool, logging_level: int):
        self.model = model
        self.stream = stream
        self.logging_level = logging_level
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.logging_level)

    @staticmethod
    def dictionarize_messages(messages: List[Message]) -> List[Dict[str, Any]]:
        return [message.to_dict() for message in messages]

    @abstractmethod
    def call_api(self, messages: List[Message], max_tokens: int) -> Dict[str, Any]:
        pass


class OllamaVendor(AiVendor):
    def __init__(
        self,
        ip_address: str,
        port: str,
        model: str,
        stream: bool = False,
        logging_level: int = logging.WARNING,
    ):
        super().__init__(model=model, stream=stream, logging_level=logging_level)
        self.ip_address = ip_address
        self.port = port
        assert stream is False, "Ollama streaming is not supported."

    def call_api(self, messages: List[Message], max_tokens: int) -> Dict[str, Any]:
        url = f"http://{self.ip_address}:{self.port}/api/chat"
        data = {
            "model": self.model,
            "messages": self.dictionarize_messages(messages),
            "stream": self.stream,
            "options": {"num_predict": max_tokens},
        }

        start_time = time.perf_counter()

        for _ in range(3):
            try:
                # if ollama is stuck, do sudo systemctl restart ollama.service
                # the try except will catch the connection error and retry
                response = requests.post(url, json=data)
                break
            except requests.exceptions.ConnectionError:
                self.logger.warning(
                    f"Connection error. Was Ollama restarted? Retrying in 5s."
                )
                time.sleep(5)

        stop_time = time.perf_counter()
        self.logger.info(
            f"Called model: {self.model} at {url}. Status: {response.status_code} "
            f"Elapsed time: {stop_time - start_time:.2f}s"
        )

        return json.loads(response.text)["message"]


class OpenAIVendor(AiVendor):
    def __init__(
        self,
        model: str,
        stream: bool = False,
        logging_level: int = logging.WARNING,
        *args,  # type: ignore
        **kwargs,  # type: ignore
    ):
        super().__init__(model=model, stream=stream, logging_level=logging_level)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key, *args, **kwargs)

    def call_api(
        self, messages: List[Message], max_tokens: int = 1000
    ) -> Dict[str, Any]:
        built_messages: List[Dict[str, Any]] = []
        for message in messages:
            built_messages.append(self._build_message(message))

        start_time = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=built_messages,  # type: ignore
            max_tokens=max_tokens,
        )
        stop_time = time.perf_counter()
        if isinstance(response.usage, CompletionUsage):
            cost = (
                response.usage.completion_tokens / 10**6 * 40
                + response.usage.prompt_tokens / 10**6 * 10
            )
        else:
            cost = -1
        self.logger.info(
            f"Called model: {self.model} at OpenAI. "
            f"Elapsed time: {stop_time - start_time:.2f}s. Usage: {response.usage} Cost: {cost:.3f}$"
        )
        return {"role": "assistant", "content": response.choices[0].message.content}

    def _build_message(self, message: Message) -> Dict[str, Any]:
        if len(message.images) > 0:
            return self._build_image_message(
                user=message.role,
                text=message.content,
                images=[message.images],
            )
        else:
            return self._build_text_message(user=message.role, text=message.content)

    def _build_text_message(self, user: str, text: str) -> Dict[str, Any]:
        return {"role": user, "content": text}

    def _build_image_message(
        self, user: str, text: str, images: List[str]
    ) -> Dict[str, Any]:
        return {
            "role": user,
            "content": [
                {"type": "text", "text": text},
            ]
            + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}",
                    },
                }
                for image in images
            ],
        }


class AWSBedrockVendor(AiVendor):
    def __init__(
        self,
        model: str,
        stream: bool = False,
        logging_level: int = logging.WARNING,
        system_prompt_allowed: bool = False,
        squash_messages: bool = True,
        *args,  # type: ignore
        **kwargs,  # type: ignore
    ):
        super().__init__(model=model, stream=stream, logging_level=logging_level)
        self.session = boto3.Session()
        self.client = self.session.client(  # type: ignore
            service_name="bedrock-runtime", region_name="us-west-2"
        )
        self.system_prompt_allowed = system_prompt_allowed
        self.squash_messages = squash_messages

    def call_api(
        self, messages: List[Dict[str, Any]], max_tokens: int = 1000
    ) -> Dict[str, Any]:
        built_messages: List[Dict[str, Any]] = []
        preprocessed_messages = self._preprocess_messages(messages)
        for message in preprocessed_messages:
            built_messages.append(self._build_message(message))

        start_time = time.perf_counter()
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": built_messages,
            }
        )
        response = self.client.invoke_model(body=body, modelId=self.model)  # type: ignore
        response_body = json.loads(response.get("body").read())  # type: ignore
        stop_time = time.perf_counter()
        self.logger.info(
            f"Called model: {self.model} at AWS Bedrock. "
            f"Elapsed time: {stop_time - start_time:.2f}s. Usage: {response_body['usage']}"
        )
        return {"role": "assistant", "content": response_body["content"][0]["text"]}

    def _preprocess_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        preprocessed_messages = []
        if not self.system_prompt_allowed:
            map_dict = {"user": "user", "system": "user", "assistant": "assistant"}
            for message in messages:
                if message.role == "system":
                    preprocessed_messages.append(
                        Message(
                            role="user", content=message.content, images=message.images
                        )
                    )
                else:
                    preprocessed_messages.append(message)

        if self.squash_messages:
            preprocessed_messages = self._squash_messages(preprocessed_messages)
        return preprocessed_messages

    def _squash_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        messages = copy.deepcopy(messages)
        new_messages: List[Dict[str, Any]] = []
        while len(messages):
            message_role = messages[0].role
            start_idx = 0
            end_idx = 0
            for idx in range(len(messages)):
                if messages[idx].role == message_role:
                    end_idx = idx
                else:
                    break

            new_message: Dict[str, Any] = {
                "role": message_role,
                "content": " ".join(
                    [message.content for message in messages[start_idx : end_idx + 1]]
                ),
                "images": [],
            }
            for i in range(start_idx, end_idx + 1):
                new_message["images"] += messages[i].images
            new_messages.append(new_message)
            messages = messages[end_idx + 1 :]

        return new_messages

    def _build_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        if len(message["images"]) > 0:
            return self._build_image_message(
                user=message["role"],
                text=message["content"],
                images=[message["images"]],
            )
        else:
            return self._build_text_message(
                user=message["role"], text=message["content"]
            )

    def _build_text_message(self, user: str, text: str) -> Dict[str, Any]:
        return {"role": user, "content": text}

    def _build_image_message(
        self, user: str, text: str, images: List[str]
    ) -> Dict[str, Any]:
        return {
            "role": user,
            "content": [
                {"type": "text", "text": text},
            ]
            + [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image[0],
                    },
                }
                for image in images
            ],
        }
