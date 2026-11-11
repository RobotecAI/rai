# Copyright (C) 2025 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""A client of a plain HTTP policy server: POST a msgpack observation, get a
msgpack dict with "actions" back. The smallest server to wrap a LeRobot, SmolVLA
or custom policy in (see the docs for a 20-line FastAPI example)."""

import asyncio
from typing import Any, Mapping, Optional

from .client import (
    ActionChunk,
    Observation,
    PolicyConnectionError,
    PolicyError,
    PolicyTimeoutError,
    as_chunk,
    require,
)
from .codec import pack_openpi, unpack_openpi

CONTENT_TYPE = "application/msgpack"


class HttpPolicyClient:
    """POST ``url`` with the observation (openpi msgpack encoding); the reply is the
    same encoding with "actions" (T, D). ``reset_url``, if set, is POSTed on reset.
    Keeps one pooled connection and reconnects on its own."""

    def __init__(
        self,
        url: str = "http://localhost:8000/infer",
        *,
        timeout_s: float = 5.0,
        headers: Optional[Mapping[str, str]] = None,
        actions_key: str = "actions",
        reset_url: Optional[str] = None,
    ) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.headers = {"Content-Type": CONTENT_TYPE, "Accept": CONTENT_TYPE}
        self.headers.update(headers or {})
        self.actions_key = actions_key
        self.reset_url = reset_url
        self._client: Any = None
        self._loop: Any = None

    async def _post(self, url: str, body: bytes) -> bytes:
        httpx = require("httpx")
        loop = asyncio.get_running_loop()
        if self._client is None or loop is not self._loop:  # clients are loop-bound
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
            self._loop = loop
        try:
            reply = await self._client.post(url, content=body, headers=self.headers)
        except httpx.TimeoutException as e:
            raise PolicyTimeoutError(
                f"The policy server at {url} did not answer within {self.timeout_s:g} s"
            ) from e
        except httpx.TransportError as e:
            raise PolicyConnectionError(
                f"Cannot reach the policy server at {url}: {e}"
            ) from e
        if reply.status_code != 200:
            raise PolicyError(
                f"The policy server at {url} answered {reply.status_code}: {reply.text[:300]}"
            )
        return reply.content

    async def infer(self, observation: Observation) -> ActionChunk:
        answer = unpack_openpi(
            await self._post(self.url, pack_openpi(dict(observation)))
        )
        if not isinstance(answer, dict) or self.actions_key not in answer:
            raise PolicyError(f"The policy reply has no {self.actions_key!r}")
        return as_chunk(answer[self.actions_key])

    async def reset(self) -> None:
        if self.reset_url:
            await self._post(self.reset_url, pack_openpi({}))

    async def close(self) -> None:
        if self._client is not None and self._loop is asyncio.get_running_loop():
            client, self._client = self._client, None
            await client.aclose()
        self._client = None
