# Copyright (C) 2026 Kajetan Rachwał
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

import asyncio
import logging
import threading
from enum import IntFlag
from typing import Callable, Optional

import aiohttp
from aiohttp import ClientSession, ClientTimeout, web


class HTTPAPIError(Exception): ...


class HTTPConnectorMode(IntFlag):
    client = 1  # 0b01
    server = 2  # 0b10
    client_server = 3  # 0b11 (client | server)


class HTTPAPI:
    def __init__(
        self,
        mode: HTTPConnectorMode = HTTPConnectorMode.client,
        host="localhost",
        port=8080,
    ):
        self.host = host
        self.port = port
        self.mode = mode

        self.routes: dict[str, list[str]] = {}

        self.app = web.Application()
        self.runner = web.AppRunner(self.app)
        self.loop = asyncio.new_event_loop()
        self.client_session = None
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._started_event = threading.Event()
        self.unresolved_futures = []

        self.websockets: dict[str, set[web.WebSocketResponse]] = {}
        self.ws_clients: dict[str, aiohttp.ClientWebSocketResponse] = {}

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._start_server())
        self._started_event.set()
        self.loop.run_forever()

    async def _start_server(self):
        if self.mode & HTTPConnectorMode.client:
            self.client_session = ClientSession()
        if self.mode & HTTPConnectorMode.server:
            await self.runner.setup()
            site = web.TCPSite(self.runner, self.host, self.port)
            await site.start()
            print(f"Serving on http://{self.host}:{self.port}")

    def run(self):
        self._thread.start()
        self._started_event.wait()

    def stop(self):
        def shutdown():
            async def _shutdown():
                if (
                    self.mode & HTTPConnectorMode.client
                    and self.client_session is not None
                ):
                    await self.client_session.close()
                if self.mode & HTTPConnectorMode.server:
                    await self.runner.cleanup()
                self.loop.stop()

            asyncio.run_coroutine_threadsafe(_shutdown(), self.loop)

        shutdown()

    def add_route(
        self,
        method: str,
        path: str,
        handler_lambda: Callable,
    ):
        if not (self.mode & HTTPConnectorMode.server):
            return

        async def handler(request):
            return await handler_lambda(request)

        def register():
            self.app.router._frozen = False
            self.app.router.add_route(method.upper(), path, handler)

        self.loop.call_soon_threadsafe(register)
        if self.routes.get(path) is None:
            self.routes[path] = [method]
        else:
            self.routes[path].append(method)

    def send_request(
        self,
        method: str,
        url: str,
        timeout: Optional[float],
        *,
        payload: Optional[str | dict],
        headers: Optional[dict],
        **kwargs,
    ) -> tuple[str, int]:
        if not (self.mode & HTTPConnectorMode.client):
            raise HTTPAPIError("Tried sending request with client mode disabled!")
        timeout_cfg = ClientTimeout(timeout) if timeout is not None else None
        if payload is not None and "json" not in kwargs and "data" not in kwargs:
            kwargs["json"] = payload  # aiohttp will set appropriate headers
        if headers is not None:
            kwargs["headers"] = {"Content-Type": "application/json"}

        if timeout_cfg:
            kwargs["timeout"] = timeout_cfg

        coro = self._request(method, url, **kwargs)
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        self.unresolved_futures.append(future)
        if timeout is None:
            return "", 200
        return future.result()

    async def _request(self, method: str, url: str, **kwargs):
        assert self.client_session is not None
        async with self.client_session.request(method.upper(), url, **kwargs) as resp:
            return await resp.text(), resp.status

    def shutdown(self):
        for future in self.unresolved_futures:
            try:
                future.result(timeout=0)
            except Exception as e:
                logging.warning(f"Background request failed or timed out: {e}")
        self.unresolved_futures.clear()
