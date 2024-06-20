# Scenario building

Scenario can consist of:

```python
ScenarioPartType = Union[
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
    ConditionalScenario,
    FutureAiMessage,
    AgentLoop,
]
```

Example scenario:

```python
    scenario: List[ScenarioPartType] = [
        SystemMessage(
            content="You are an autonomous agent. Your main goal is to fulfill the user's requests. "
            "Do not make assumptions about the environment you are currently in. "
            "Use the tooling provided to gather information about the environment."
            "You are always required to send a voice message to the user about your decisions. This is crucial."
            "The voice message should contain a very short information about what is going on and what is the next step. "
        ),
        HumanMessage(
            content="The robot is moving. Use vision to understand the surroundings, and add waypoints based on observations. camera is accesible at topic /camera/camera/color/image_raw ."
        ),
        AgentLoop(stop_action=FinishTool().__class__.__name__, stop_iters=50),
    ]
```
