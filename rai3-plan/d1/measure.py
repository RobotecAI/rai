"""D1 measurements. Run from rai3-plan/: python -m d1.measure  (writes d1/measurements.json)."""

from __future__ import annotations

import asyncio
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .common.guard import Recorder
from .common.robot import FakeRobot, observe_status
from .common.script import Call, Script, Text, read_loop

HERE = Path(__file__).parent
SLICE_FILES = {
    "langchain": ["slice.py", "mcp.py"],
    "pydanticai": ["slice.py", "mcp.py"],
}


def loc(path: Path) -> int:
    """Non-blank lines that are not pure comments."""
    lines = path.read_text().splitlines()
    return sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#"))


def import_time(stmt: str, runs: int = 9) -> float:
    code = (
        f"import time; t = time.perf_counter(); {stmt}; print(time.perf_counter() - t)"
    )
    out = [
        float(
            subprocess.run(
                [sys.executable, "-c", code],
                cwd=HERE.parent,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        for _ in range(runs)
    ]
    return statistics.median(out)


class Timed(Script):
    def next(self, messages):
        self.stamps.append(time.perf_counter())
        return super().next(messages)


def timed(n: int) -> Timed:
    s = Timed(read_loop(n).steps)
    s.stamps = []
    return s


def step_latency(fw: str, full: bool, n: int = 300, reps: int = 3) -> float:
    """Median wall time of one ReAct step (model call + READ tool call) with the fake model."""
    samples = []
    for _ in range(reps):
        io, script = FakeRobot(), timed(n)
        kw = (
            dict(hooks=[Recorder()], context=lambda: observe_status(io)) if full else {}
        )
        kw["approve_actuation"] = full
        if fw == "langchain":
            from .langchain.fake import ScriptedChatModel
            from .langchain.slice import build_agent, make_tools

            agent = build_agent(ScriptedChatModel(script=script), make_tools(io), **kw)
            cfg = {"configurable": {"thread_id": "b"}, "recursion_limit": 10 * n}
            asyncio.run(agent.ainvoke({"messages": [("user", "go")]}, cfg))
        else:
            from pydantic_ai.usage import UsageLimits

            from .pydanticai.fake import scripted_model
            from .pydanticai.slice import build_agent, make_tools

            agent = build_agent(scripted_model(script), [make_tools(io)], **kw)
            asyncio.run(agent.run("go", usage_limits=UsageLimits(request_limit=None)))
        st = script.stamps
        samples += [b - a for a, b in zip(st[10:], st[11:])]  # skip warm-up steps
    return statistics.median(samples)


def history_growth(fw: str, n: int = 300) -> float:
    """Step latency at the end of an n-step run divided by latency at the start."""
    io, script = FakeRobot(), timed(n)
    if fw == "langchain":
        from .langchain.fake import ScriptedChatModel
        from .langchain.slice import build_agent, make_tools

        agent = build_agent(
            ScriptedChatModel(script=script), make_tools(io), approve_actuation=False
        )
        cfg = {"configurable": {"thread_id": "b"}, "recursion_limit": 10 * n}
        asyncio.run(agent.ainvoke({"messages": [("user", "go")]}, cfg))
    else:
        from pydantic_ai.usage import UsageLimits

        from .pydanticai.fake import scripted_model
        from .pydanticai.slice import build_agent, make_tools

        agent = build_agent(
            scripted_model(script), [make_tools(io)], approve_actuation=False
        )
        asyncio.run(agent.run("go", usage_limits=UsageLimits(request_limit=None)))
    d = [b - a for a, b in zip(script.stamps, script.stamps[1:])]
    return statistics.median(d[-30:]) / statistics.median(d[10:40])


def cancel_latency(fw: str, reps: int = 20) -> float:
    slow = Call("publish_twist", {"linear_x": 0.3, "angular_z": 0.0, "duration_s": 10})
    out = []
    for _ in range(reps):
        io = FakeRobot()
        script = Script([[slow], Text("done")])
        if fw == "langchain":
            from .langchain.fake import ScriptedChatModel
            from .langchain.slice import build_agent, make_tools

            agent = build_agent(
                ScriptedChatModel(script=script),
                make_tools(io),
                approve_actuation=False,
            )
            coro = lambda: agent.ainvoke(
                {"messages": [("user", "go")]}, {"configurable": {"thread_id": "c"}}
            )
        else:
            from .pydanticai.fake import scripted_model
            from .pydanticai.slice import build_agent, make_tools

            agent = build_agent(
                scripted_model(script), [make_tools(io)], approve_actuation=False
            )
            coro = lambda: agent.run("go")

        async def main():
            task = asyncio.create_task(coro())
            while not io.published:
                await asyncio.sleep(0.001)
            await asyncio.sleep(0.05)
            t0 = time.perf_counter()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            assert io.published[-1] == (0.0, 0.0)
            return time.perf_counter() - t0

        out.append(asyncio.run(main()))
    return statistics.median(out)


def main() -> None:
    r: dict = {"python": sys.version.split()[0]}
    from importlib.metadata import version

    r["versions"] = {
        p: version(p)
        for p in [
            "langchain",
            "langchain-core",
            "langgraph",
            "langchain-mcp-adapters",
            "pydantic-ai-slim",
            "mcp",
            "fastmcp-slim",
        ]
    }
    for fw, files in SLICE_FILES.items():
        r[f"{fw}.loc"] = {f: loc(HERE / fw / f) for f in files}
        r[f"{fw}.loc_total"] = sum(r[f"{fw}.loc"].values())
    r["common.loc"] = {
        f.name: loc(f)
        for f in sorted((HERE / "common").glob("*.py"))
        if f.name != "__init__.py"
    }
    r["import.langchain.agents"] = import_time("import langchain.agents")
    r["import.pydantic_ai"] = import_time("import pydantic_ai")
    r["import.langchain.slice+mcp"] = import_time(
        "import d1.langchain.slice, d1.langchain.mcp"
    )
    r["import.pydanticai.slice+mcp"] = import_time(
        "import d1.pydanticai.slice, d1.pydanticai.mcp"
    )
    for fw in SLICE_FILES:
        r[f"{fw}.step_ms.bare"] = 1e3 * step_latency(fw, full=False)
        r[f"{fw}.step_ms.full"] = 1e3 * step_latency(fw, full=True)
        r[f"{fw}.step_growth_300"] = history_growth(fw)
        r[f"{fw}.cancel_ms"] = 1e3 * cancel_latency(fw)
    (HERE / "measurements.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
