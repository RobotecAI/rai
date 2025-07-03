import asyncio
import threading
from enum import IntFlag

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
        self.app = web.Application()
        self.runner = web.AppRunner(self.app)
        self.loop = asyncio.new_event_loop()
        self.client_session = None
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._started_event = threading.Event()

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

    def add_route(self, method: str, path: str, handler_lambda):
        if not (self.mode & HTTPConnectorMode.server):
            return

        async def handler(request):
            return await handler_lambda(request)

        def register():
            self.app.router._frozen = False
            self.app.router.add_route(method.upper(), path, handler)

        self.loop.call_soon_threadsafe(register)

    def add_websocket(self, path: str, handler_lambda):
        if not (self.mode & HTTPConnectorMode.server):
            return

        async def ws_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await handler_lambda(ws, request)
            return ws

        def register():
            self.app.router.add_get(path, ws_handler)

        self.loop.call_soon_threadsafe(register)

    def send_request(
        self, method: str, url: str, timeout: float, **kwargs
    ) -> tuple[str, int]:
        if not (self.mode & HTTPConnectorMode.client):
            raise HTTPAPIError("Tried sending request with client mode disabled!")
        timeout_cfg = ClientTimeout(timeout)
        coro = self._request(method, url, timeout=timeout_cfg, **kwargs)
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    async def _request(self, method: str, url: str, **kwargs):
        assert self.client_session is not None
        async with self.client_session.request(method.upper(), url, **kwargs) as resp:
            return await resp.text(), resp.status
