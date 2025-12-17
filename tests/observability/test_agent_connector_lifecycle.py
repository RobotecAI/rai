from rai.agents.base import BaseAgent
from rai.communication.base_connector import BaseConnector, BaseMessage


class ListSink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record(self, event: dict) -> None:
        self.events.append(event)


class DummyAgent(BaseAgent):
    def run(self):
        return None

    def stop(self):
        return None


class DummyMessage(BaseMessage):
    pass


class DummyConnector(BaseConnector[DummyMessage]):
    pass


def test_base_agent_emits_agent_start_and_sets_sink(monkeypatch):
    # Keep heartbeat disabled for a deterministic unit test.
    monkeypatch.delenv("RAI_OBS_HEARTBEAT_SEC", raising=False)
    sink = ListSink()

    agent = DummyAgent(name="a1", observability_sink=sink)

    assert agent.observability_sink is sink
    assert any(ev.get("event_type") == "agent_start" for ev in sink.events)
    start_events = [ev for ev in sink.events if ev.get("event_type") == "agent_start"]
    assert len(start_events) == 1
    assert start_events[0]["component_type"] == "agent"
    assert start_events[0]["agent_name"] == "a1"


def test_connector_open_close_events_via_attach_and_shutdown(monkeypatch):
    monkeypatch.delenv("RAI_OBS_HEARTBEAT_SEC", raising=False)
    sink = ListSink()

    conn = DummyConnector(observability_sink=sink)
    conn.attach_observability(agent_name="a1", sink=sink)
    conn.shutdown()

    types = [ev.get("event_type") for ev in sink.events]
    assert "connector_open" in types
    assert "connector_close" in types

    open_events = [ev for ev in sink.events if ev.get("event_type") == "connector_open"]
    close_events = [
        ev for ev in sink.events if ev.get("event_type") == "connector_close"
    ]
    assert len(open_events) == 1
    assert len(close_events) == 1

    assert open_events[0]["component_type"] == "connector"
    assert close_events[0]["component_type"] == "connector"
    assert open_events[0]["agent_name"] == "a1"
    assert close_events[0]["agent_name"] == "a1"


def test_attach_connectors_wires_observability_sink(monkeypatch):
    monkeypatch.delenv("RAI_OBS_HEARTBEAT_SEC", raising=False)
    sink = ListSink()
    agent = DummyAgent(name="a1", observability_sink=sink)

    conn = DummyConnector(observability_sink=sink)
    agent.attach_connectors(conn)

    assert conn.agent_name == "a1"
    assert conn.observability_sink is sink
