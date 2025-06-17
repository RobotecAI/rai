import asyncio
import logging

from aiohttp import ClientSession, web


class HTTPAPI:
    def __init__(self, host="localhost", port=8080):
        self.app = web.Application()
        self.host = host
        self.port = port
        self.runner = web.AppRunner(self.app)
        self.loop = asyncio.get_event_loop()
        self.client_session = ClientSession()

    def add_route(self, method: str, path: str, handler_lambda):
        async def handler(request):
            return await handler_lambda(request)

        self.app.router.add_route(method.upper(), path, handler)

    def add_websocket(self, path: str, handler_lambda):
        async def ws_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await handler_lambda(ws, request)
            return ws

        self.app.router.add_get(path, ws_handler)

    # ----------- HTTP Client Methods ------------
    async def request(self, method: str, url: str, **kwargs):
        async with self.client_session.request(method.upper(), url, **kwargs) as resp:
            return await resp.text(), resp.status

    async def get(self, url: str, **kwargs):
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs):
        return await self.request("POST", url, **kwargs)

    def run(self):
        self.loop.run_until_complete(self._start())

    async def _start(self):
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        logging.info(f"Serving on http://{self.host}:{self.port}")
        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            await self.client_session.close()
