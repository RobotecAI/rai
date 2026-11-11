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


"""A client of openpi's websocket policy server (``scripts/serve_policy.py``).

Protocol (openpi_client.websocket_client_policy): the server sends its metadata
first; then each binary request frame is a msgpack observation, each binary reply
an action dict with "actions" (T, D). A text frame is a server traceback, after
which the server closes the connection.
"""

import asyncio
from typing import Any, Optional

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


class OpenPIClient:
    """openpi websocket policy client (pi0, pi0-FAST, pi0.5 and any server that
    speaks the protocol). Connects lazily, reconnects after errors and cancels."""

    def __init__(
        self,
        url: str = "ws://localhost:8000",
        *,
        api_key: Optional[str] = None,
        timeout_s: float = 5.0,
        connect_timeout_s: float = 5.0,
        actions_key: str = "actions",
    ) -> None:
        self.url = url if url.startswith(("ws://", "wss://")) else f"ws://{url}"
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.actions_key = actions_key
        self.metadata: dict[str, Any] = {}
        self.server_timing: dict[str, Any] = {}
        self._ws: Any = None
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _bind(self) -> None:
        """Connections and locks belong to one event loop; start over on another."""
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            if (
                self._ws is not None
                and self._loop is not None
                and not self._loop.is_closed()
            ):
                self._loop.call_soon_threadsafe(self._ws.transport.abort)
            self._ws, self._lock, self._loop = None, asyncio.Lock(), loop

    async def _connect(self) -> Any:
        client = require("websockets.asyncio.client")
        exceptions = require("websockets.exceptions")
        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
        try:
            async with asyncio.timeout(self.connect_timeout_s):
                ws = await client.connect(
                    self.url,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=None,
                    close_timeout=1.0,
                )
                try:
                    self.metadata = unpack_openpi(await ws.recv()) or {}
                except BaseException:
                    ws.transport.abort()
                    raise
        except TimeoutError as e:
            raise PolicyConnectionError(
                f"No openpi policy server at {self.url} "
                f"(no handshake within {self.connect_timeout_s:g} s)"
            ) from e
        except (OSError, exceptions.WebSocketException) as e:
            raise PolicyConnectionError(
                f"Cannot connect to the openpi policy server at {self.url}: {e}"
            ) from e
        return ws

    def _drop(self) -> None:
        if self._ws is not None:
            self._ws.transport.abort()  # immediate; a late reply must never be read
            self._ws = None

    async def infer(self, observation: Observation) -> ActionChunk:
        exceptions = require("websockets.exceptions")
        request = pack_openpi(dict(observation))
        self._bind()
        async with self._lock:
            for retry in (False, True):
                fresh = self._ws is None
                if fresh:
                    self._ws = await self._connect()
                try:
                    async with asyncio.timeout(self.timeout_s):
                        await self._ws.send(request)
                        reply = await self._ws.recv()
                except exceptions.ConnectionClosed as e:
                    self._drop()
                    if fresh or retry:
                        raise PolicyConnectionError(
                            f"The openpi policy server at {self.url} closed the connection: {e}"
                        ) from e
                    continue  # a stale connection: reconnect once
                except TimeoutError as e:
                    self._drop()
                    raise PolicyTimeoutError(
                        f"The openpi policy server at {self.url} did not answer "
                        f"within {self.timeout_s:g} s"
                    ) from e
                except BaseException:  # cancelled: the reply would be out of step
                    self._drop()
                    raise
                break
            else:  # the second attempt breaks or raises
                raise AssertionError("unreachable")
        if isinstance(reply, str):
            self._drop()
            raise PolicyError(f"openpi policy server error: {_tail(reply)}")
        answer = unpack_openpi(reply)
        if not isinstance(answer, dict) or self.actions_key not in answer:
            raise PolicyError(f"The openpi reply has no {self.actions_key!r}")
        self.server_timing = answer.get("server_timing") or {}
        return as_chunk(answer[self.actions_key])

    async def reset(self) -> None:
        """openpi has no reset message; the policy is stateless between calls."""

    async def close(self) -> None:
        self._bind()
        async with self._lock:
            ws, self._ws = self._ws, None
        if ws is not None:
            await ws.close()


def _tail(traceback: str, lines: int = 3) -> str:
    return " | ".join(line.strip() for line in traceback.strip().splitlines()[-lines:])
