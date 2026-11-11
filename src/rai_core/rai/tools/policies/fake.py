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


"""A scripted policy server that speaks openpi's websocket protocol, for tests and
for dry runs of a policy skill's ROS wiring without a GPU."""

import asyncio
import contextlib
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import numpy as np

from .client import Observation, require
from .codec import pack_openpi, unpack_openpi

Script = Union[Sequence[Any], Callable[[Observation], Any]]


class FakePolicyServer:
    """Answers each observation with the next scripted action chunk (the last one
    repeats), or with ``script(observation)``. Runs on its own thread::

        with FakePolicyServer([np.zeros((5, 7))]) as server:
            client = OpenPIClient(server.url)

    ``observations`` records what it received; ``delay_s`` delays each answer;
    a script entry that is an Exception is sent back as a server error.
    """

    def __init__(
        self,
        script: Script,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        delay_s: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.script = script
        self.host = host
        self.port = port
        self.delay_s = delay_s
        self.metadata = metadata or {}
        self.observations: list[dict[str, Any]] = []
        self.request_times: list[float] = []
        self.connections = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop: Optional[asyncio.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def _answer(self, observation: Observation) -> Any:
        if callable(self.script):
            return self.script(observation)
        index = min(len(self.observations) - 1, len(self.script) - 1)
        return self.script[index]

    async def _handle(self, ws: Any) -> None:
        self.connections += 1
        exceptions = require("websockets.exceptions")
        await ws.send(pack_openpi(self.metadata))
        try:
            async for frame in ws:
                observation = unpack_openpi(frame)
                self.observations.append(observation)
                self.request_times.append(time.monotonic())
                if self.delay_s and self._stop is not None:
                    with contextlib.suppress(TimeoutError):  # wake up on stop()
                        await asyncio.wait_for(self._stop.wait(), self.delay_s)
                answer = self._answer(observation)
                if isinstance(answer, BaseException):
                    await ws.send(f"Traceback (most recent call last):\n{answer!r}")
                    await ws.close(1011, "Internal server error")
                    return
                await ws.send(
                    pack_openpi({"actions": np.asarray(answer), "server_timing": {}})
                )
        except exceptions.ConnectionClosed:
            pass

    async def _serve(self) -> None:
        server_module = require("websockets.asyncio.server")
        self._stop = asyncio.Event()
        try:
            async with server_module.serve(
                self._handle, self.host, self.port, compression=None, max_size=None
            ) as server:
                self.port = server.sockets[0].getsockname()[1]
                self._ready.set()
                await self._stop.wait()
                for connection in list(server.connections):
                    connection.transport.abort()  # a stopped server, not a polite one
        except BaseException as e:
            self._error = e
            self._ready.set()

    def start(self) -> "FakePolicyServer":
        def run() -> None:
            self._loop = asyncio.new_event_loop()
            self._loop.run_until_complete(self._serve())
            self._loop.close()

        self._thread = threading.Thread(target=run, name="fake-policy", daemon=True)
        self._thread.start()
        self._ready.wait(10.0)
        if self._error is not None:
            raise self._error
        return self

    def stop(self) -> None:
        if self._loop is not None and self._stop is not None and self._thread:
            self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(5.0)
            self._thread = None

    def __enter__(self) -> "FakePolicyServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()
