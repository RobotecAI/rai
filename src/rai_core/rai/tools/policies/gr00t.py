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


"""A client of Isaac GR00T's policy server (``gr00t/eval/run_gr00t_server.py``).

Protocol (gr00t/policy/server_client.py, N1.6 and N1.7): ZMQ REQ/REP, msgpack
requests ``{"endpoint", "data", "api_token"}``, arrays as np.save bytes. Errors
come back as ``{"error": str}``.
"""

import asyncio
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from .client import (
    ActionChunk,
    Observation,
    PolicyError,
    PolicyTimeoutError,
    as_chunk,
    require,
)
from .codec import pack_gr00t, unpack_gr00t

MODALITIES = ("video", "state", "language")


def nest(observation: Observation) -> dict[str, dict[str, Any]]:
    """Flat "video.<key>", "state.<key>", "language.<key>" entries to GR00T's nested
    observation with batch and time axes of 1: video uint8 (1, 1, H, W, 3), state
    float32 (1, 1, D), language [[text]]."""
    nested: dict[str, dict[str, Any]] = {m: {} for m in MODALITIES}
    for name, value in observation.items():
        modality, _, key = name.partition(".")
        if modality not in nested or not key:
            raise ValueError(
                f"GR00T observation key {name!r} must start with one of "
                f"{', '.join(m + '.' for m in MODALITIES)}"
            )
        if modality == "language":
            nested[modality][key] = [[str(value)]]
        elif modality == "video":
            nested[modality][key] = np.asarray(value, dtype=np.uint8)[None, None]
        else:
            nested[modality][key] = np.asarray(value, dtype=np.float32)[None, None]
    return {m: v for m, v in nested.items() if v}


class Gr00tClient:
    """Isaac GR00T N1.x policy client. Observation keys are "video.<key>",
    "state.<key>" and "language.<key>" with the checkpoint's modality keys (ask
    the server with `modality_config()`). The action chunk is the action keys
    concatenated in `action_keys` order (the server's order if None)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        *,
        timeout_s: float = 5.0,
        api_token: Optional[str] = None,
        action_keys: Optional[Sequence[str]] = None,
    ) -> None:
        self.address = f"tcp://{host}:{port}"
        self.timeout_s = timeout_s
        self.api_token = api_token
        self.action_keys = list(action_keys) if action_keys else None
        self._socket: Any = None
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _bind(self) -> None:
        """Sockets and locks belong to one event loop; start over on another."""
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            self._drop()
            self._lock, self._loop = asyncio.Lock(), loop

    def _drop(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)  # a REQ socket is stuck after a lost reply
            self._socket = None

    async def call(self, endpoint: str, data: Optional[dict[str, Any]] = None) -> Any:
        """Call one server endpoint ("ping", "get_action", "reset", ...)."""
        zmq = require("zmq")
        zmq_asyncio = require("zmq.asyncio")
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token
        payload = pack_gr00t(request)
        self._bind()
        async with self._lock:
            if self._socket is None:
                self._socket = zmq_asyncio.Context.instance().socket(zmq.REQ)
                self._socket.connect(self.address)
            try:
                async with asyncio.timeout(self.timeout_s):
                    await self._socket.send(payload)
                    reply = await self._socket.recv()
            except TimeoutError as e:
                self._drop()
                raise PolicyTimeoutError(
                    f"The GR00T policy server at {self.address} did not answer "
                    f"{endpoint} within {self.timeout_s:g} s (is it running?)"
                ) from e
            except BaseException:
                self._drop()
                raise
        if reply == b"ERROR":
            raise PolicyError(f"The GR00T policy server at {self.address} failed")
        answer = unpack_gr00t(reply)
        if isinstance(answer, dict) and "error" in answer:
            raise PolicyError(f"GR00T policy server error: {answer['error']}")
        return answer

    async def ping(self) -> bool:
        return (await self.call("ping") or {}).get("status") == "ok"

    async def modality_config(self) -> Any:
        return await self.call("get_modality_config")

    async def infer(self, observation: Observation) -> ActionChunk:
        answer = await self.call(
            "get_action", {"observation": nest(observation), "options": None}
        )
        actions = answer[0] if isinstance(answer, list | tuple) else answer
        if not isinstance(actions, dict) or not actions:
            raise PolicyError("The GR00T reply has no action dict")
        keys = self.action_keys or list(actions)
        missing = [k for k in keys if k not in actions]
        if missing:
            raise PolicyError(
                f"The GR00T reply has no {missing}; it has {list(actions)}"
            )
        parts = []
        for key in keys:
            part = np.asarray(actions[key], dtype=np.float64)
            if part.ndim == 3:  # (B, T, D)
                part = part[0]
            if part.ndim == 1:  # (T,)
                part = part[:, None]
            parts.append(part)
        try:
            return as_chunk(np.concatenate(parts, axis=-1))
        except ValueError as e:
            raise PolicyError(f"GR00T action keys do not line up: {e}") from e

    async def reset(self) -> None:
        await self.call("reset", {"options": None})

    async def close(self) -> None:
        self._bind()
        async with self._lock:
            self._drop()
