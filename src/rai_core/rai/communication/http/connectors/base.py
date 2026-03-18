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

from typing import Any, Callable, Dict, Optional, TypeVar

from aiohttp import web

from rai.communication import BaseConnector
from rai.communication.http.api import HTTPAPI, HTTPConnectorMode
from rai.communication.http.messages import HTTPMessage

T = TypeVar("T", bound=HTTPMessage)


class HTTPBaseConnector(BaseConnector[T]):
    """HTTP-based implementation of BaseConnector.

    Supports client (send/poll), server (receive callbacks), or combined modes.

    Parameters
    ----------
    host : str
        Hostname for the HTTP server (server/client_server mode).
    port : int
        Port for the HTTP server (server/client_server mode).
    mode : HTTPConnectorMode
        Operating mode: client, server, or client_server.
    """

    def __init__(
        self,
        host: str,
        port: int,
        mode: HTTPConnectorMode = HTTPConnectorMode.client,
    ):
        super().__init__()
        self.api = HTTPAPI(mode, host, port)
        self.api.run()
        self._last_msg: Dict[str, T] = {}

    def service_call(
        self,
        message: T,
        target: str,
        timeout_sec: float,
        **kwargs: Optional[Any],
    ) -> T:
        """Send an HTTP request and wait for the response (request-response pattern).

        Parameters
        ----------
        message : T
            Request message; ``message.method`` determines the HTTP verb.
        target : str
            Full URL of the service endpoint.
        timeout_sec : float
            Timeout in seconds.
        """
        ret, code = self.api.send_request(
            method=message.method,
            url=target,
            timeout=timeout_sec,
            payload=message.payload,
            headers=message.metadata.get("headers"),
        )
        return self.T_class(
            payload=ret, method=message.method, metadata={"return_code": code}
        )  # type: ignore[call-arg]

    def create_service(
        self,
        service_name: str,
        on_request: Callable,
        on_done: Optional[Callable] = None,
        *,
        method: str | None = "POST",
        **kwargs: Optional[Any],
    ) -> str:
        """Create an HTTP endpoint that handles request-response interactions.

        Parameters
        ----------
        service_name : str
            URL path of the service (e.g. ``"/api/chat"``).
        on_request : Callable
            Called with the parsed request body; its return value is sent back.
        on_done : Callable, optional
            Currently unused; reserved for future lifecycle hooks.
        method : str | None
            HTTP method to listen for, will default to ``"POST"``.

        Returns
        -------
        str
            The service path (``service_name``).
        """

        if method is None:
            method = "POST"

        async def route_handler(request: web.Request) -> web.Response:
            try:
                data = await request.json()
            except Exception:
                data = await request.text()
            result = on_request(data)
            if result is None:
                return web.Response(status=200)
            if isinstance(result, dict):
                return web.json_response(result)
            return web.Response(text=str(result), status=200)

        self.api.add_route(method, service_name, route_handler)
        return service_name

    def shutdown(self) -> None:
        """Stop the HTTP API and release all resources."""
        self.api.stop()
        super().shutdown()


class HTTPConnector(HTTPBaseConnector[HTTPMessage]):
    """Concrete HTTP connector for plain :class:`HTTPMessage` exchange."""
