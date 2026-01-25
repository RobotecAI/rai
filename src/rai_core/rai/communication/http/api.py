import asyncio
import threading
from enum import IntFlag
from typing import Callable, Optional, Any
import json
import logging

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
        if self.routes.get(path) is not None:
            self.routes[path] = [method]
        else:
            self.routes[path].append(method)

    def add_websocket(self, path: str, handler_lambda):
        """
        In server mode:
            `path` is the HTTP path (e.g. "/ws").
            `handler_lambda(ws, request)` is called for each connection.

        In client mode:
            `path` is the full WebSocket URL (e.g. "ws://example.com/ws").
            `handler_lambda(ws, msg)` is called for each incoming message.
        """
        # SERVER SIDE
        if self.mode & HTTPConnectorMode.server:
            if path not in self.websockets:
                self.websockets[path] = set()

            async def ws_handler(request):
                ws = web.WebSocketResponse()
                await ws.prepare(request)

                # register this connection
                self.websockets[path].add(ws)

                try:
                    # user handler can read/write freely, e.g.:
                    # async for msg in ws: ...
                    await handler_lambda(ws, request)
                finally:
                    # ensure it is removed on close
                    self.websockets[path].discard(ws)
                    await ws.close()

                return ws

            def register_server():
                self.app.router.add_get(path, ws_handler)

            self.loop.call_soon_threadsafe(register_server)

        # CLIENT SIDE
        if self.mode & HTTPConnectorMode.client:
            async def connect_client_ws():
                assert self.client_session is not None, "ClientSession not initialized"
                ws = await self.client_session.ws_connect(path)
                self.ws_clients[path] = ws

                try:
                    async for msg in ws:
                        # let user handler inspect/read messages and optionally write
                        await handler_lambda(ws, msg)
                finally:
                    # clean up on close
                    if self.ws_clients.get(path) is ws:
                        del self.ws_clients[path]
                    await ws.close()

            def start_client():
                asyncio.create_task(connect_client_ws())

            self.loop.call_soon_threadsafe(start_client)

    def publish_websocket(
            self,
            path: str,
            payload: Optional[str | dict],
    ):
        """
        Send `payload` over all WebSocket connections associated with `path`.

        - For server mode: broadcasts to all connected clients on that route.
        - For client mode: sends to the single client WebSocket created for that URL.
        """
        if payload is None:
            msg = ""
        elif isinstance(payload, dict):
            msg = json.dumps(payload)
        else:
            msg = str(payload)

        async def _publish():
            # collect all websockets (server + client) associated with this key
            server_conns = list(self.websockets.get(path, []))
            client_ws = self.ws_clients.get(path)
            all_conns = server_conns + ([client_ws] if client_ws is not None else [])

            dead_server = []
            dead_client = False

            for ws in all_conns:
                try:
                    await ws.send_str(msg)
                except Exception:
                    # mark broken ones to be removed
                    if ws in server_conns:
                        dead_server.append(ws)
                    elif ws is client_ws:
                        dead_client = True

            # cleanup broken server connections
            for ws in dead_server:
                self.websockets.get(path, set()).discard(ws)

            # cleanup broken client connection
            if dead_client and self.ws_clients.get(path) is client_ws:
                del self.ws_clients[path]

        asyncio.run_coroutine_threadsafe(_publish(), self.loop)

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
