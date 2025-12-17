import time

from rai.observability.correlation import observability_context
from rai.observability.lifecycle import emit_event, heartbeat_period_s, start_heartbeat


class ListSink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record(self, event: dict) -> None:
        self.events.append(event)


def test_heartbeat_period_s_env_gate(monkeypatch):
    monkeypatch.delenv("RAI_OBS_HEARTBEAT_SEC", raising=False)
    assert heartbeat_period_s() is None

    monkeypatch.setenv("RAI_OBS_HEARTBEAT_SEC", "0")
    assert heartbeat_period_s() is None

    monkeypatch.setenv("RAI_OBS_HEARTBEAT_SEC", "-1")
    assert heartbeat_period_s() is None

    monkeypatch.setenv("RAI_OBS_HEARTBEAT_SEC", "not-a-number")
    assert heartbeat_period_s() is None

    monkeypatch.setenv("RAI_OBS_HEARTBEAT_SEC", "0.5")
    assert heartbeat_period_s() == 0.5


def test_emit_event_includes_correlation_context():
    sink = ListSink()
    with observability_context(run_id="r1", job_id="j1", task_id="t1", request_id="q1"):
        emit_event(
            sink=sink,
            event_type="agent_start",
            component_type="agent",
            agent_name="a",
        )

    assert len(sink.events) == 1
    ev = sink.events[0]
    assert ev["event_type"] == "agent_start"
    assert ev["component_type"] == "agent"
    assert ev["agent_name"] == "a"
    assert ev["run_id"] == "r1"
    assert ev["job_id"] == "j1"
    assert ev["task_id"] == "t1"
    assert ev["request_id"] == "q1"


def test_start_heartbeat_opt_in_and_stoppable(monkeypatch):
    # Opt-in via env.
    monkeypatch.setenv("RAI_OBS_HEARTBEAT_SEC", "0.01")
    sink = ListSink()

    hb = start_heartbeat(
        sink=sink,
        component_type="agent",
        agent_name="agent",
    )
    assert hb is not None

    # Wait until we see at least two heartbeats (first should be immediate-ish).
    deadline = time.time() + 1.0
    while time.time() < deadline and len(sink.events) < 2:
        time.sleep(0.005)

    hb.stop()

    assert len(sink.events) >= 1
    assert all(ev["event_type"] == "heartbeat" for ev in sink.events)
    assert all(ev["component_type"] == "agent" for ev in sink.events)
    assert all(ev["agent_name"] == "agent" for ev in sink.events)


def test_start_heartbeat_disabled_returns_none(monkeypatch):
    monkeypatch.delenv("RAI_OBS_HEARTBEAT_SEC", raising=False)
    sink = ListSink()
    hb = start_heartbeat(sink=sink, component_type="agent", agent_name="agent")
    assert hb is None
    assert sink.events == []
