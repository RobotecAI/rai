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


"""Policy server clients against in-process fake servers: openpi (websocket),
GR00T (ZMQ) and HTTP; timeouts, reconnects, server errors and cancellation."""

import asyncio
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pytest
from rai.tools.policies import (
    FakePolicyServer,
    Gr00tClient,
    HttpPolicyClient,
    OpenPIClient,
    PolicyClient,
    PolicyConnectionError,
    PolicyError,
    PolicyTimeoutError,
)
from rai.tools.policies.client import as_chunk, require
from rai.tools.policies.codec import (
    pack_gr00t,
    pack_openpi,
    unpack_gr00t,
    unpack_openpi,
)
from rai.tools.policies.gr00t import nest

pytestmark = pytest.mark.timeout(60)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------ codec


def test_openpi_codec_roundtrip() -> None:
    obs = {
        "observation/image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "observation/state": np.array([0.5, -1.0], dtype=np.float32),
        "scalar": np.float32(2.5),  # np.float64 is a float: packed natively
        "prompt": "pick",
    }
    back = unpack_openpi(pack_openpi(obs))
    assert back["prompt"] == "pick"
    assert back["scalar"] == 2.5 and isinstance(back["scalar"], np.float32)
    np.testing.assert_array_equal(back["observation/image"], obs["observation/image"])
    assert back["observation/state"].dtype == np.float32
    with pytest.raises(ValueError):
        pack_openpi({"bad": np.array([object()])})


def test_openpi_codec_matches_openpi_wire_format() -> None:
    """Bytes keys, dtype.str and the shape, as openpi_client.msgpack_numpy writes."""
    msgpack = require("msgpack")
    raw = msgpack.unpackb(pack_openpi({"a": np.zeros((2, 3), "<f4")}))
    assert raw["a"][b"__ndarray__"] is True
    assert raw["a"][b"dtype"] == "<f4" and raw["a"][b"shape"] == [2, 3]
    assert len(raw["a"][b"data"]) == 24


def test_gr00t_codec_roundtrip_and_msgpack_numpy() -> None:
    msgpack = require("msgpack")
    data = {"endpoint": "get_action", "data": {"x": np.ones((1, 2, 3), np.float32)}}
    back = unpack_gr00t(pack_gr00t(data))
    np.testing.assert_array_equal(back["data"]["x"], data["data"]["x"])
    # msgpack_numpy's own encoding, which N1.7 servers may send
    arr = np.arange(6, dtype="<f8").reshape(2, 3)
    wire = msgpack.packb(
        {
            "a": {
                b"nd": True,
                b"type": "<f8",
                b"kind": b"",
                b"shape": [2, 3],
                b"data": arr.tobytes(),
            }
        }
    )
    np.testing.assert_array_equal(unpack_gr00t(wire)["a"], arr)


def test_as_chunk_shapes_and_values() -> None:
    assert as_chunk([1.0, 2.0]).shape == (1, 2)
    assert as_chunk(np.zeros((1, 4, 3))).shape == (4, 3)
    with pytest.raises(PolicyError, match="NaN"):
        as_chunk([[np.nan]])
    with pytest.raises(PolicyError, match="shape"):
        as_chunk(np.zeros((2, 2, 2)))


def test_gr00t_nest() -> None:
    nested = nest(
        {
            "video.ego_view": np.zeros((4, 5, 3), np.uint8),
            "state.arm": [1.0, 2.0],
            "language.annotation.human.task_description": "wave",
        }
    )
    assert nested["video"]["ego_view"].shape == (1, 1, 4, 5, 3)
    assert nested["state"]["arm"].dtype == np.float32
    assert nested["state"]["arm"].shape == (1, 1, 2)
    assert nested["language"]["annotation.human.task_description"] == [["wave"]]
    with pytest.raises(ValueError, match="must start with"):
        nest({"prompt": "x"})


def test_clients_are_policy_clients() -> None:
    for client in (OpenPIClient(), Gr00tClient(), HttpPolicyClient()):
        assert isinstance(client, PolicyClient)


# ------------------------------------------------------------------ openpi


def test_openpi_infer_against_fake_server() -> None:
    chunks = [np.full((3, 2), 1.0), np.full((3, 2), 2.0)]
    with FakePolicyServer(chunks, metadata={"action_horizon": 3}) as server:

        async def run() -> list[Any]:
            client = OpenPIClient(server.url)
            try:
                first = await client.infer({"prompt": "go", "x": np.ones(2)})
                second = await client.infer({"prompt": "go", "x": np.ones(2)})
                third = await client.infer({"prompt": "go", "x": np.ones(2)})
                return [first, second, third, client.metadata]
            finally:
                await client.close()

        first, second, third, metadata = asyncio.run(run())
    assert first.shape == (3, 2) and first[0, 0] == 1.0
    assert second[0, 0] == 2.0 and third[0, 0] == 2.0  # the last chunk repeats
    assert metadata == {"action_horizon": 3}
    assert server.observations[0]["prompt"] == "go"
    assert server.connections == 1  # one connection for every call


def test_openpi_server_down_is_a_connection_error() -> None:
    client = OpenPIClient(f"ws://127.0.0.1:{free_port()}", connect_timeout_s=1.0)
    with pytest.raises(PolicyConnectionError, match="Cannot connect"):
        asyncio.run(client.infer({"x": np.ones(1)}))


def test_openpi_timeout_then_reconnect() -> None:
    with FakePolicyServer([np.zeros((1, 1))], delay_s=0.5) as server:

        async def run() -> Any:
            client = OpenPIClient(server.url, timeout_s=0.1)
            with pytest.raises(PolicyTimeoutError):
                await client.infer({"x": np.ones(1)})
            client.timeout_s = 5.0
            chunk = await client.infer({"x": np.ones(1)})
            await client.close()
            return chunk

        assert asyncio.run(run()).shape == (1, 1)
    assert server.connections == 2  # the timed-out connection was dropped


def test_openpi_server_error_and_restart() -> None:
    script = [np.zeros((1, 2)), RuntimeError("model exploded"), np.ones((1, 2))]
    with FakePolicyServer(script) as server:

        async def run() -> Any:
            client = OpenPIClient(server.url)
            await client.infer({"x": np.ones(1)})
            with pytest.raises(PolicyError, match="model exploded"):
                await client.infer({"x": np.ones(1)})
            chunk = await client.infer({"x": np.ones(1)})  # reconnects
            await client.close()
            return chunk

        assert asyncio.run(run())[0, 0] == 1.0
    assert server.connections == 2


def test_openpi_reconnects_after_server_restart() -> None:
    port = free_port()

    async def infer(client: OpenPIClient) -> Any:
        return await client.infer({"x": np.ones(1)})

    client = OpenPIClient(f"ws://127.0.0.1:{port}")
    loop = asyncio.new_event_loop()
    try:
        with FakePolicyServer([np.zeros((1, 1))], port=port):
            loop.run_until_complete(infer(client))
        with FakePolicyServer([np.ones((1, 1))], port=port) as second:
            assert loop.run_until_complete(infer(client))[0, 0] == 1.0
        assert second.connections == 1
    finally:
        loop.run_until_complete(client.close())
        loop.close()


def test_openpi_cancel_is_prompt_and_resyncs() -> None:
    with FakePolicyServer([np.zeros((1, 1)), np.ones((1, 1))], delay_s=0.3) as server:

        async def run() -> tuple[float, Any]:
            client = OpenPIClient(server.url)
            task = asyncio.ensure_future(client.infer({"x": np.ones(1)}))
            await asyncio.sleep(0.1)
            start = time.monotonic()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            latency = time.monotonic() - start
            chunk = await client.infer({"x": np.ones(1)})  # not the stale reply
            await client.close()
            return latency, chunk

        latency, chunk = asyncio.run(run())
    assert latency < 0.05
    assert chunk[0, 0] == 1.0  # the second answer; the first was never read


def test_client_survives_a_new_event_loop() -> None:
    """RaiTool._run runs on a fresh loop per call when called from a loop thread."""
    with FakePolicyServer([np.zeros((1, 1))]) as server:
        client = OpenPIClient(server.url)
        asyncio.run(client.infer({"x": np.ones(1)}))
        asyncio.run(client.infer({"x": np.ones(1)}))
        asyncio.run(client.close())
    assert server.connections == 2


# ------------------------------------------------------------------ GR00T


class FakeGr00tServer:
    """GR00T's PolicyServer loop: ZMQ REP, msgpack, {"error"} on failure."""

    def __init__(self, delay_s: float = 0.0, token: str | None = None) -> None:
        zmq = require("zmq")
        self.delay_s, self.token = delay_s, token
        self.requests: list[dict[str, Any]] = []
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def handle(self, request: dict[str, Any]) -> Any:
        if self.token and request.get("api_token") != self.token:
            return {"error": "Unauthorized: Invalid API token"}
        endpoint = request.get("endpoint", "get_action")
        if endpoint == "ping":
            return {"status": "ok", "message": "Server is running"}
        if endpoint == "reset":
            return {}
        if endpoint == "get_action":
            obs = request["data"]["observation"]
            assert obs["video"]["cam"].shape[:2] == (1, 1)
            arm = np.tile(obs["state"]["arm"][:, :, :2], (1, 4, 1)).astype(np.float32)
            gripper = np.full((1, 4, 1), 0.5, np.float32)
            return [{"arm": arm, "gripper": gripper}, {}]
        return {"error": f"Unknown endpoint: {endpoint}"}

    def run(self) -> None:
        zmq = require("zmq")
        while self.running:
            if not self.socket.poll(50):
                continue
            request = unpack_gr00t(self.socket.recv())
            self.requests.append(request)
            time.sleep(self.delay_s)
            try:
                reply = self.handle(request)
            except Exception as e:
                reply = {"error": str(e)}
            self.socket.send(pack_gr00t(reply))
        self.socket.close(linger=0)
        self.context.term()
        del zmq

    def stop(self) -> None:
        self.running = False
        self.thread.join(5.0)


@pytest.fixture
def gr00t() -> Iterator[FakeGr00tServer]:
    server = FakeGr00tServer(token="secret")
    yield server
    server.stop()


def test_gr00t_infer(gr00t: FakeGr00tServer) -> None:
    async def run() -> Any:
        client = Gr00tClient("127.0.0.1", gr00t.port, api_token="secret")
        try:
            assert await client.ping()
            await client.reset()
            return await client.infer(
                {
                    "video.cam": np.zeros((8, 8, 3), np.uint8),
                    "state.arm": np.array([0.1, 0.2, 0.3]),
                    "language.task": "wave",
                }
            )
        finally:
            await client.close()

    chunk = asyncio.run(run())
    assert chunk.shape == (4, 3)
    np.testing.assert_allclose(chunk[0], [0.1, 0.2, 0.5], rtol=1e-6)
    assert gr00t.requests[-1]["data"]["observation"]["language"]["task"] == [["wave"]]


def test_gr00t_errors(gr00t: FakeGr00tServer) -> None:
    async def run() -> None:
        client = Gr00tClient("127.0.0.1", gr00t.port, api_token="wrong")
        with pytest.raises(PolicyError, match="Unauthorized"):
            await client.ping()
        client.api_token = "secret"
        client.action_keys = ["arm", "missing"]
        with pytest.raises(PolicyError, match="missing"):
            await client.infer({"video.cam": np.zeros((2, 2, 3)), "state.arm": [0, 1]})
        await client.close()

    asyncio.run(run())


def test_gr00t_server_down_times_out_then_recovers() -> None:
    port = free_port()

    async def run() -> None:
        client = Gr00tClient("127.0.0.1", port, timeout_s=0.2)
        with pytest.raises(PolicyTimeoutError, match="is it running"):
            await client.ping()
        server = FakeGr00tServer()
        try:
            client.address = f"tcp://127.0.0.1:{server.port}"
            assert await client.ping()  # a fresh REQ socket after the lost reply
        finally:
            server.stop()
            await client.close()

    asyncio.run(run())


# ------------------------------------------------------------------ HTTP


@pytest.fixture
def http_server() -> Iterator[tuple[str, list[Any]]]:
    seen: list[Any] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/broken":
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"CUDA out of memory")
                return
            obs = unpack_openpi(body)
            seen.append((self.path, obs))
            if self.path == "/slow":
                time.sleep(0.5)
            reply = pack_openpi({"actions": np.full((2, 3), 7.0, np.float32)})
            self.send_response(200)
            self.send_header("Content-Type", "application/msgpack")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            try:
                self.wfile.write(reply)
            except BrokenPipeError:  # the client timed out
                pass

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", seen
    server.shutdown()
    server.server_close()


def test_http_client(http_server: tuple[str, list[Any]]) -> None:
    url, seen = http_server

    async def run() -> Any:
        client = HttpPolicyClient(f"{url}/infer", reset_url=f"{url}/reset")
        try:
            await client.reset()
            return await client.infer({"state": np.ones(2), "prompt": "go"})
        finally:
            await client.close()

    chunk = asyncio.run(run())
    assert chunk.shape == (2, 3) and chunk[0, 0] == 7.0
    assert [path for path, _ in seen] == ["/reset", "/infer"]
    assert seen[1][1]["prompt"] == "go"


def test_http_errors(http_server: tuple[str, list[Any]]) -> None:
    url, _ = http_server

    async def run() -> None:
        with pytest.raises(PolicyError, match="500: CUDA out of memory"):
            await HttpPolicyClient(f"{url}/broken").infer({})
        with pytest.raises(PolicyTimeoutError):
            await HttpPolicyClient(f"{url}/slow", timeout_s=0.1).infer({})
        down = HttpPolicyClient(f"http://127.0.0.1:{free_port()}/infer")
        with pytest.raises(PolicyConnectionError):
            await down.infer({})

    asyncio.run(run())


def test_missing_extra_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "msgpack", None)
    with pytest.raises(ImportError, match=r"rai_core\[policies\]"):
        pack_openpi({})
