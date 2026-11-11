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

"""ROS2Connector benchmark: throughput, latency, leaks and shutdown behavior.

Usage (with ROS 2 and the RAI env sourced):
    python tests/benchmarks/ros2_connector_bench.py --out bench.json
    python tests/benchmarks/ros2_connector_bench.py --quick \\
        --out new.json --compare baseline.json
or through pytest (deselected by default): pytest -m benchmark tests/benchmarks -s

Services, actions and topic streams are served by subprocesses of this script so
that discovery and transport costs are real. Every benchmark is isolated: a
failure or hang is recorded as {"error": ...} and the run continues.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

SRV_A, SRV_B, ACTION = "/bench_srv_a", "/bench_srv_b", "/bench_fib"
PUB_SMALL, PUB_IMG, PUB_HRI = "/bench_pub_small", "/bench_pub_img", "/bench_pub_hri"
RECV_TOPICS = [f"/bench_recv_{i}" for i in range(10)]
STRING, TRIGGER = "std_msgs/msg/String", "std_srvs/srv/Trigger"
FIBONACCI = "example_interfaces/action/Fibonacci"


# ------------------------------------------------------------ subprocesses


def serve() -> None:
    """Services, an action, receive topics at 20 Hz and sinks for publish benches."""
    import rclpy
    from example_interfaces.action import Fibonacci
    from rclpy.action import ActionServer
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    from rai_interfaces.msg import HRIMessage

    rclpy.init()
    node = rclpy.create_node("rai_bench_server")
    group = ReentrantCallbackGroup()

    def trigger(request, response):
        response.success = True
        return response

    def execute(goal_handle):
        for i in range(goal_handle.request.order):
            feedback = Fibonacci.Feedback()
            feedback.sequence = [i]
            goal_handle.publish_feedback(feedback)
            time.sleep(0.005)
        goal_handle.succeed()
        return Fibonacci.Result()

    for name in (SRV_A, SRV_B):
        node.create_service(Trigger, name, trigger, callback_group=group)
    ActionServer(node, Fibonacci, ACTION, execute, callback_group=group)
    for msg_cls, topic in (
        (String, PUB_SMALL),
        (Image, PUB_IMG),
        (HRIMessage, PUB_HRI),
    ):
        node.create_subscription(msg_cls, topic, lambda _: None, 10)
    publishers = [node.create_publisher(String, t, 10) for t in RECV_TOPICS]
    node.create_timer(0.05, lambda: [p.publish(String(data="x")) for p in publishers])
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()


CLOUD_POINTS = 100_000  # x, y, z, intensity as float32: 1.6 MB, a LiDAR sweep


def stream_message(kind: str) -> tuple[type, Callable[[str], Any]]:
    """Message class and a factory stamping "<seq> <monotonic_ns>" into it."""
    from sensor_msgs.msg import Image, PointCloud2, PointField
    from std_msgs.msg import String

    from rai_interfaces.msg import HRIMessage

    image = Image(height=480, width=640, encoding="rgb8", step=640 * 3)
    image.data = bytes(640 * 480 * 3)
    if kind == "string":
        return String, lambda text: String(data=text)
    if kind == "hri":
        return HRIMessage, lambda text: HRIMessage(text=text, images=[image])
    if kind == "image":
        msg = image
    else:
        fields = [
            PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
            for i, n in enumerate(("x", "y", "z", "intensity"))
        ]
        msg = PointCloud2(height=1, width=CLOUD_POINTS, point_step=16, fields=fields)
        msg.row_step = 16 * CLOUD_POINTS
        msg.data = bytes(16 * CLOUD_POINTS)

    def make(text: str) -> Any:
        msg.header.frame_id = text
        return msg

    return type(msg), make


def stream(topic: str, rate: float, count: int, kind: str = "string") -> None:
    """Publish `count` messages of `kind` at `rate` Hz once subscribed."""
    import rclpy

    rclpy.init()
    node = rclpy.create_node("rai_bench_stream")
    msg_cls, make = stream_message(kind)
    publisher = node.create_publisher(msg_cls, topic, 10)

    deadline = time.monotonic() + 10.0
    while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)
    start = time.monotonic()
    for seq in range(count):
        delay = start + seq / rate - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        publisher.publish(make(f"{seq} {time.monotonic_ns()}"))
    time.sleep(0.2)


def spawn(*args: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, os.path.abspath(__file__), *args])


# ------------------------------------------------------------------ helpers


def stats(seconds: list[float]) -> dict[str, Any]:
    if not seconds:
        return {"n": 0}
    xs = sorted(seconds)
    return {
        "n": len(xs),
        "p50_ms": round(xs[len(xs) // 2] * 1e3, 3),
        "p99_ms": round(xs[min(len(xs) - 1, int(len(xs) * 0.99))] * 1e3, 3),
    }


def time_calls(fns: list[Callable[[], Any]]) -> dict[str, Any]:
    durations = []
    start = time.perf_counter()
    for fn in fns:
        t0 = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - t0)
    return {
        **stats(durations),
        "per_s": round(len(fns) / (time.perf_counter() - start), 1),
    }


def close(connector: Any, timeout: float = 5.0) -> float:
    """Shut a connector down without letting a hang block the benchmark."""
    t0 = time.perf_counter()
    thread = threading.Thread(target=connector.close, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"close() did not return in {timeout}s")
    return time.perf_counter() - t0


def entity_counts(connector: Any) -> dict[str, Any]:
    node = connector._node
    with open("/proc/self/status") as f:
        rss_kb = next(int(line.split()[1]) for line in f if line.startswith("VmRSS"))
    return {
        "subscriptions": len(list(node.subscriptions)),
        "publishers": len(list(node.publishers)),
        "clients": len(list(node.clients)),
        "waitables": len(list(node.waitables)),
        "threads": threading.active_count(),
        "rss_kb": rss_kb,
    }


def connector_bench(
    fn: Callable[[Any], dict[str, Any]], hri: bool = False
) -> Callable[[], dict]:
    def run() -> dict[str, Any]:
        from rai.communication.ros2 import ROS2Connector, ROS2HRIConnector

        connector = ROS2HRIConnector() if hri else ROS2Connector()
        try:
            return fn(connector)
        finally:
            close(connector)

    return run


# --------------------------------------------------------------- benchmarks


def bench_publish(n: int) -> dict[str, Any]:
    import numpy as np
    from rai.communication import HRIMessage
    from rai.communication.ros2 import ROS2Connector, ROS2HRIConnector, ROS2Message
    from rai.content import Image

    out: dict[str, Any] = {}
    connector = ROS2Connector()
    try:
        small = ROS2Message(payload={"data": "hello"})
        out["string"] = time_calls(
            [lambda: connector.send_message(small, PUB_SMALL, msg_type=STRING)] * n
        )
        image = {
            "height": 480,
            "width": 640,
            "encoding": "rgb8",
            "step": 640 * 3,
            "data": bytes(640 * 480 * 3),
        }
        out["image_vga"] = time_calls(
            [
                lambda: connector.send_message(
                    ROS2Message(payload=image),
                    PUB_IMG,
                    msg_type="sensor_msgs/msg/Image",
                )
            ]
            * max(n // 20, 5)
        )
    finally:
        close(connector)
    hri = ROS2HRIConnector()
    try:
        message = HRIMessage("hi", [Image(np.zeros((480, 640, 3), np.uint8))])
        out["hri_vga"] = time_calls(
            [lambda: hri.sync.send(PUB_HRI, message)] * max(n // 40, 5)
        )
    finally:
        close(hri)
    return out


def numpy_view(msg: Any) -> Any:
    """Zero-copy numpy view of an Image or PointCloud2 (3.0: rai views)."""
    import numpy as np

    if hasattr(msg, "encoding"):
        return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
    dtype = np.dtype(
        {
            "names": [f.name for f in msg.fields],
            "formats": ["<f4"] * len(msg.fields),
            "offsets": [f.offset for f in msg.fields],
            "itemsize": msg.point_step,
        }
    )
    return np.frombuffer(msg.data, dtype)


def bench_dispatch(
    rate: float, seconds: float, kind: str = "string"
) -> Callable[[Any], dict[str, Any]]:
    """Stream to a registered callback: Strings to a raw callback, VGA
    HRIMessages to a decoded one, VGA Images and PointCloud2s to the default
    (wrapped) callback that takes a numpy view of the data."""

    def run(connector: Any) -> dict[str, Any]:
        topic = f"/bench_dispatch_{kind}_{int(rate)}"
        count = int(rate * seconds)
        received: list[tuple] = []
        views: list[float] = []
        lock = threading.Lock()

        def callback(msg: Any) -> None:
            now = time.monotonic_ns()
            if kind in ("image", "cloud"):
                t0 = time.perf_counter()
                numpy_view(msg.payload)
                views.append(time.perf_counter() - t0)
                text = msg.payload.header.frame_id
            else:
                text = msg.text if kind == "hri" else msg.data
            seq, sent = map(int, text.split())
            with lock:
                received.append((seq, (now - sent) / 1e9))

        if kind == "string":
            connector.register_callback(topic, callback, raw=True, msg_type=STRING)
        elif kind == "hri":
            connector.subscribe(topic, callback)
        else:
            msg_type = {
                "image": "sensor_msgs/msg/Image",
                "cloud": "sensor_msgs/msg/PointCloud2",
            }[kind]
            connector.register_callback(topic, callback, msg_type=msg_type)
        publisher = spawn("--stream", topic, str(rate), str(count), kind)
        try:
            publisher.wait(timeout=seconds + 15)
        finally:
            publisher.kill()
        time.sleep(1.0)
        with lock:
            seqs = [seq for seq, _ in received]
            latencies = [latency for _, latency in received]
        return {
            "latency": stats(latencies),
            "sent": count,
            "delivered": len(seqs),
            "dropped": count - len(set(seqs)),
            "inversions": sum(1 for a, b in zip(seqs, seqs[1:]) if b < a),
            **({"numpy_view": stats(views)} if views else {}),
        }

    return run


def bench_receive(n_warm: int) -> Callable[[Any], dict[str, Any]]:
    def run(connector: Any) -> dict[str, Any]:
        cold = time_calls(
            [
                lambda t=t: connector.receive_message(
                    t, timeout_sec=5.0, msg_type=STRING
                )
                for t in RECV_TOPICS
            ]
        )
        warm = time_calls(
            [lambda: connector.receive_message(RECV_TOPICS[0], timeout_sec=1.0)]
            * n_warm
        )
        return {"cold": cold, "warm": warm}

    return run


def call_trigger(connector: Any, name: str, reuse_client: bool = True) -> None:
    from rai.communication.ros2 import ROS2Message

    connector.service_call(
        ROS2Message(payload={}),
        name,
        timeout_sec=2.0,
        msg_type=TRIGGER,
        reuse_client=reuse_client,
    )


def bench_service(n: int) -> Callable[[Any], dict[str, Any]]:
    def run(connector: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out["first_call"] = time_calls([lambda: call_trigger(connector, SRV_A)])
        out["reused"] = time_calls([lambda: call_trigger(connector, SRV_A)] * n)
        out["fresh"] = time_calls(
            [lambda: call_trigger(connector, SRV_A, reuse_client=False)]
            * max(n // 10, 5)
        )

        latencies: list[float] = []
        errors: list[str] = []
        lock = threading.Lock()
        per_thread = max(n // 8, 5)

        def worker(name: str) -> None:
            for _ in range(per_thread):
                t0 = time.perf_counter()
                try:
                    call_trigger(connector, name)
                except Exception as e:
                    with lock:
                        errors.append(f"{type(e).__name__}: {e}")
                    continue
                with lock:
                    latencies.append(time.perf_counter() - t0)

        threads = [
            threading.Thread(target=worker, args=(SRV_A if i % 2 else SRV_B,))
            for i in range(8)
        ]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - start
        out["concurrent_8x2"] = {
            **stats(latencies),
            "per_s": round(len(latencies) / wall, 1),
            "errors": len(errors),
            "first_error": errors[0] if errors else None,
        }
        return out

    return run


def run_goal(connector: Any, timeout: float = 5.0) -> dict[str, float]:
    from rai.communication.ros2 import ROS2Message

    t0 = time.perf_counter()
    marks: dict[str, float] = {}
    done = threading.Event()

    def on_feedback(_: Any) -> None:
        marks.setdefault("first_feedback", time.perf_counter() - t0)

    def on_done(_: Any) -> None:
        marks["result"] = time.perf_counter() - t0
        done.set()

    connector.start_action(
        ROS2Message(payload={"order": 3}),
        ACTION,
        on_feedback=on_feedback,
        on_done=on_done,
        timeout_sec=timeout,
        msg_type=FIBONACCI,
    )
    marks["accepted"] = time.perf_counter() - t0
    if not done.wait(timeout):
        raise TimeoutError("action result not received")
    return marks


def bench_action(n: int) -> Callable[[Any], dict[str, Any]]:
    def run(connector: Any) -> dict[str, Any]:
        marks = [run_goal(connector) for _ in range(n)]
        return {
            key: stats([m[key] for m in marks if key in m])
            for key in ("accepted", "first_feedback", "result")
        }

    return run


def count_errors(fn: Callable[[], Any], n: int, budget_s: float) -> dict[str, Any]:
    """Run fn up to n times within budget_s; leaks show up as errors or slowdown."""
    errors = []
    deadline = time.monotonic() + budget_s
    done = 0
    while done < n and time.monotonic() < deadline:
        done += 1
        try:
            fn()
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
    return {"done": done, "errors": len(errors), "first": errors[0] if errors else None}


def bench_leak(n_calls: int, n_actions: int, n_register: int):
    def run(connector: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"before": entity_counts(connector)}
        t0 = time.perf_counter()
        out["service_call_errors"] = count_errors(
            lambda: call_trigger(connector, SRV_A, reuse_client=False), n_calls, 30.0
        )
        out["after_service_calls"] = entity_counts(connector)
        out["service_calls_s"] = round(time.perf_counter() - t0, 2)

        t0 = time.perf_counter()
        out["action_errors"] = count_errors(
            lambda: run_goal(connector), n_actions, 30.0
        )
        out["after_actions"] = entity_counts(connector)
        out["actions_s"] = round(time.perf_counter() - t0, 2)

        t0 = time.perf_counter()
        for _ in range(n_register):
            callback_id = connector.register_callback(
                RECV_TOPICS[0], lambda _: None, msg_type=STRING
            )
            connector.unregister_callback(callback_id)
        out["after_register"] = entity_counts(connector)
        out["register_s"] = round(time.perf_counter() - t0, 2)
        out["register_unregister"] = n_register
        return out

    return run


def bench_hooks(n: int) -> Callable[[Any], dict[str, Any]]:
    """Publish cost per call (us) with no egress hook, a hook that does not
    match, one matching allow hook (sync, async) and one that grants a lease.
    send_message is the sync path; publish the async one, timed in one loop."""

    def per_call_us(durations: list[float]) -> dict[str, Any]:
        xs = sorted(durations)
        return {
            "p50_us": round(xs[len(xs) // 2] * 1e6, 2),
            "mean_us": round(sum(xs) / len(xs) * 1e6, 2),
        }

    def run(connector: Any) -> dict[str, Any]:
        import asyncio

        from rai.communication.ros2 import ROS2Message
        from std_msgs.msg import String

        msg = ROS2Message(payload={"data": "hello"})
        ros_msg = String(data="hello")

        def sync_path() -> dict[str, Any]:
            durations = []
            for _ in range(n):
                t0 = time.perf_counter()
                connector.send_message(msg, PUB_SMALL, msg_type=STRING)
                durations.append(time.perf_counter() - t0)
            return per_call_us(durations)

        async def async_loop() -> list[float]:
            durations = []
            for _ in range(n):
                t0 = time.perf_counter()
                await connector.publish(PUB_SMALL, ros_msg)
                durations.append(time.perf_counter() - t0)
            return durations

        def async_path() -> dict[str, Any]:
            return per_call_us(asyncio.run(async_loop()))

        def measure() -> dict[str, Any]:
            sync_path()  # warm up the publisher and the match cache
            return {"sync": sync_path(), "async": async_path()}

        out: dict[str, Any] = {"off": measure()}
        if not hasattr(connector, "egress"):
            return {**out, "skipped": "no egress hook API (R6)"}
        from rai.communication.ros2.egress import MOTION, Allow

        def allow(egress: Any) -> Any:
            return Allow()

        async def allow_async(egress: Any) -> Any:
            return Allow()

        def lease(egress: Any) -> Any:
            return Allow(lease_s=3600.0)

        for row, hook, types in (
            ("unmatched", allow, MOTION),
            ("on", allow, {STRING}),
            ("on_async", allow_async, {STRING}),
            ("lease", lease, {STRING}),
        ):
            connector.egress.add(hook, types=types)
            out[row] = measure()
            connector.egress.remove(hook)
        return out

    return run


def bench_shutdown() -> dict[str, Any]:
    from rai.communication.ros2 import ROS2Connector

    """Duration of shutdown() while slow callbacks run and more are queued."""
    connector = ROS2Connector()
    topic = "/bench_shutdown"
    calls: list[float] = []

    def slow_callback(_: Any) -> None:
        time.sleep(0.05)
        calls.append(time.monotonic())

    connector.register_callback(topic, slow_callback, raw=True, msg_type=STRING)
    publisher = spawn("--stream", topic, "100", "300")
    try:
        deadline = time.monotonic() + 10.0
        while len(calls) < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
        duration = close(connector)
    finally:
        publisher.kill()
    return {"duration_ms": round(duration * 1e3, 3)}


# --------------------------------------------------------------------- main


def run_isolated(fn: Callable[[], dict[str, Any]], timeout: float) -> dict[str, Any]:
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = fn()
        except BaseException as e:
            result["value"] = {"error": f"{type(e).__name__}: {e}"}

    thread = threading.Thread(target=target, daemon=True)
    start = time.perf_counter()
    thread.start()
    thread.join(timeout)
    value = result.get("value", {"error": f"timeout after {timeout}s"})
    value["elapsed_s"] = round(time.perf_counter() - start, 2)
    return value


def wait_for_server() -> None:
    from rai.communication.ros2 import ROS2Connector

    connector = ROS2Connector()
    try:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            names = {name for name, _ in connector.get_services_names_and_types()}
            if {SRV_A, SRV_B} <= names:
                return
            time.sleep(0.1)
        raise TimeoutError("bench server did not come up")
    finally:
        close(connector)


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{path}."))
        elif not isinstance(value, bool):
            out[path] = value
    return out


def compare(baseline: dict[str, Any], current: dict[str, Any]) -> None:
    base = flatten(baseline["results"])
    new = flatten(current["results"])
    print(f"{'metric':58} {'baseline':>12} {'current':>12} {'delta':>9}")
    for key in sorted(base.keys() | new.keys()):
        if key.endswith(("elapsed_s", ".n")):
            continue
        a, b = base.get(key), new.get(key)
        if a == b and not key.endswith("error"):
            continue
        delta = ""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a:
            delta = f"{(b - a) / abs(a) * 100:+.0f}%"
        print(f"{key:58} {str(a)[:12]:>12} {str(b)[:12]:>12} {delta:>9}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--executor", choices=["single", "events"])
    args = parser.parse_args()
    if args.executor:  # the default executor of every runtime in this process
        from rai.communication.ros2 import runtime

        runtime.DEFAULT_EXECUTOR = args.executor
    k = 0.2 if args.quick else 1.0
    dispatch_s = 1.0 if args.quick else 2.0

    import rclpy

    rclpy.init()
    server = spawn("--server")
    started = time.perf_counter()
    results: dict[str, Any] = {}
    try:
        wait_for_server()
        benches: dict[str, tuple] = {
            "publish": (lambda: bench_publish(int(2000 * k)), 60),
            "dispatch_100hz": (connector_bench(bench_dispatch(100, dispatch_s)), 30),
            "dispatch_1khz": (connector_bench(bench_dispatch(1000, dispatch_s)), 30),
            "dispatch_hri_30hz": (
                connector_bench(bench_dispatch(30, dispatch_s, "hri"), hri=True),
                30,
            ),
            "dispatch_image_vga_30hz": (
                connector_bench(bench_dispatch(30, dispatch_s, "image")),
                30,
            ),
            "dispatch_pointcloud_10hz": (
                connector_bench(bench_dispatch(10, 2 * dispatch_s, "cloud")),
                30,
            ),
            "receive_message": (connector_bench(bench_receive(int(200 * k))), 30),
            "service_call": (connector_bench(bench_service(int(200 * k))), 60),
            "start_action": (connector_bench(bench_action(int(20 * k))), 60),
            "leak": (
                connector_bench(bench_leak(int(1000 * k), int(200 * k), int(1000 * k))),
                90,
            ),
            "hooks": (connector_bench(bench_hooks(int(2000 * k))), 60),
            "shutdown": (bench_shutdown, 30),
        }
        for name, (fn, timeout) in benches.items():
            results[name] = run_isolated(fn, timeout)
            print(f"{name}: {json.dumps(results[name])}", flush=True)
    finally:
        server.kill()
    report = {
        "meta": {
            "ros_distro": os.environ.get("ROS_DISTRO"),
            "rmw": os.environ.get("RMW_IMPLEMENTATION", "default"),
            "quick": args.quick,
            "executor": args.executor or "default",
            "total_s": round(time.perf_counter() - started, 1),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out} in {report['meta']['total_s']}s", flush=True)
    if args.compare:
        with open(args.compare) as f:
            compare(json.load(f), report)
    sys.stdout.flush()
    os._exit(0)  # connector worker threads of hung benchmarks must not block exit


if __name__ == "__main__":
    if sys.argv[1:2] == ["--server"]:
        serve()
    elif sys.argv[1:2] == ["--stream"]:
        stream(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), *sys.argv[5:6])
    else:
        main()
