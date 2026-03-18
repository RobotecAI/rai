import base64
import threading
import time
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from PIL import Image

from rai.communication.http.api import HTTPConnectorMode
from rai.communication.http.connectors import HTTPConnector, HTTPHRIConnector
from rai.communication.http.messages import HTTPHRIMessage, HTTPMessage


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_image() -> Image.Image:
    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    return img


def _image_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client_connector():
    c = HTTPConnector(host="localhost", port=19080, mode=HTTPConnectorMode.client)
    yield c
    c.shutdown()


@pytest.fixture
def server_connector():
    c = HTTPConnector(host="localhost", port=19081, mode=HTTPConnectorMode.server)
    yield c
    c.shutdown()


@pytest.fixture
def cs_connector():
    """Client-server connector, useful for end-to-end tests."""
    c = HTTPConnector(host="localhost", port=19082, mode=HTTPConnectorMode.client_server)
    yield c
    c.shutdown()


@pytest.fixture
def hri_client_connector():
    c = HTTPHRIConnector(host="localhost", port=19083, mode=HTTPConnectorMode.client)
    yield c
    c.shutdown()


@pytest.fixture
def hri_cs_connector():
    c = HTTPHRIConnector(host="localhost", port=19084, mode=HTTPConnectorMode.client_server)
    yield c
    c.shutdown()


# ── HTTPConnector – initialisation ────────────────────────────────────────────

class TestHTTPConnectorInit:
    def test_api_started(self, client_connector):
        assert client_connector.api._started_event.is_set()

    def test_t_class_is_http_message(self, client_connector):
        assert client_connector.T_class is HTTPMessage

    def test_thread_alive(self, client_connector):
        assert client_connector.api._thread.is_alive()


# ── HTTPConnector – service_call (request-response) ───────────────────────────

class TestHTTPConnectorServiceCall:
    def test_service_call_returns_http_message(self, cs_connector):
        async def handler(request):
            return web.Response(text="pong", status=200)

        cs_connector.api.add_route("GET", "/ping", handler)
        time.sleep(0.1)

        msg = HTTPMessage(method="GET", payload=None)
        response = cs_connector.service_call(msg, "http://localhost:19082/ping", timeout_sec=5.0)

        assert isinstance(response, HTTPMessage)
        assert response.payload == "pong"
        assert response.metadata["return_code"] == 200

    def test_service_call_sends_payload(self, cs_connector):
        received_body = []

        async def handler(request):
            body = await request.json()
            received_body.append(body)
            return web.Response(text="ok")

        cs_connector.api.add_route("POST", "/echo", handler)
        time.sleep(0.1)

        msg = HTTPMessage(method="POST", payload={"key": "val"})
        cs_connector.service_call(msg, "http://localhost:19082/echo", timeout_sec=5.0)

        assert received_body[0] == {"key": "val"}

    def test_service_call_preserves_method(self, cs_connector):
        async def handler(request):
            return web.Response(text="ok")

        cs_connector.api.add_route("PUT", "/put", handler)
        time.sleep(0.1)

        msg = HTTPMessage(method="PUT", payload=None)
        response = cs_connector.service_call(msg, "http://localhost:19082/put", timeout_sec=5.0)
        assert response.method == "PUT"


# ── HTTPConnector – create_service ────────────────────────────────────────────

class TestHTTPConnectorCreateService:
    def test_create_service_returns_path(self, cs_connector):
        handle = cs_connector.create_service("/svc", lambda data: None)
        assert handle == "/svc"

    def test_create_service_calls_on_request(self, cs_connector):
        calls = []

        def handler(data):
            calls.append(data)
            return {"status": "ok"}

        cs_connector.create_service("/process", handler, method="POST")
        time.sleep(0.1)

        body, code = cs_connector.api.send_request(
            "POST",
            "http://localhost:19082/process",
            timeout=5.0,
            payload={"input": 1},
            headers=None,
        )

        assert code == 200
        assert calls[0] == {"input": 1}

    def test_create_service_returns_dict_as_json(self, cs_connector):
        import json

        cs_connector.create_service("/dict_svc", lambda d: {"echo": d}, method="POST")
        time.sleep(0.1)

        body, code = cs_connector.api.send_request(
            "POST",
            "http://localhost:19082/dict_svc",
            timeout=5.0,
            payload={"x": 7},
            headers=None,
        )
        assert code == 200
        assert json.loads(body)["echo"] == {"x": 7}

    def test_create_service_none_result_returns_200(self, cs_connector):
        cs_connector.create_service("/void", lambda d: None, method="POST")
        time.sleep(0.1)

        _, code = cs_connector.api.send_request(
            "POST",
            "http://localhost:19082/void",
            timeout=5.0,
            payload={},
            headers=None,
        )
        assert code == 200


# ── HTTPConnector – shutdown ───────────────────────────────────────────────────

class TestHTTPConnectorShutdown:
    def test_shutdown_stops_event_loop(self):
        c = HTTPConnector(host="localhost", port=19090, mode=HTTPConnectorMode.client)
        c.shutdown()
        time.sleep(0.1)
        assert not c.api.loop.is_running()


# ── HTTPHRIConnector – build_message (HRIConnector mixin) ────────────────────

class TestHTTPHRIConnectorBuildMessage:
    def test_build_message_from_human_message(self, hri_client_connector):
        from langchain_core.messages import HumanMessage

        lc_msg = HumanMessage(content="hello")
        hri_msg = hri_client_connector.build_message(lc_msg)

        assert isinstance(hri_msg, HTTPHRIMessage)
        assert hri_msg.text == "hello"
        assert hri_msg.message_author == "human"

    def test_build_message_from_ai_message(self, hri_client_connector):
        from langchain_core.messages import AIMessage

        lc_msg = AIMessage(content="I am a robot")
        hri_msg = hri_client_connector.build_message(lc_msg)

        assert isinstance(hri_msg, HTTPHRIMessage)
        assert hri_msg.text == "I am a robot"
        assert hri_msg.message_author == "ai"
