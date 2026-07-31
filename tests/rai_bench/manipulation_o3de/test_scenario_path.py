# Copyright (C) 2026
import ast
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "src/rai_bench/rai_bench/manipulation_o3de/benchmark.py"


def _load_scenario_class():
    """Load Scenario from benchmark.py without importing heavy ROS deps."""
    src = BENCH.read_text()
    # Parse and exec only the Scenario class body is hard; instead stub deps then import.
    stubs = [
        "rai",
        "rai.agents",
        "rai.agents.langchain",
        "rai.messages",
        "rai.types",
        "rai_sim",
        "rai_sim.simulation_bridge",
        "langchain_core",
        "langchain_core.language_models",
        "langchain_core.messages",
        "langchain_core.runnables",
        "langchain_core.tools",
        "rai_bench",
        "rai_bench.agents",
        "rai_bench.manipulation_o3de",
        "rai_bench.manipulation_o3de.interfaces",
        "rai_bench.utils",
        "rai_bench.results_tracking",
    ]
    for name in stubs:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []  # type: ignore
            sys.modules[name] = m
    # Extra symbols used at import
    sys.modules["rai.types"].Header = object
    sys.modules["rai.types"].Point = object
    sys.modules["rai.types"].Pose = object
    sys.modules["rai.types"].PoseStamped = object
    sys.modules["rai.types"].Quaternion = object
    sys.modules["rai.agents.langchain"].ReActAgent = object
    sys.modules["rai.messages"].HumanMultimodalMessage = object
    sys.modules["langchain_core.language_models"].BaseChatModel = object
    sys.modules["langchain_core.messages"].BaseMessage = object
    sys.modules["langchain_core.runnables"].Runnable = object
    sys.modules["langchain_core.tools"].BaseTool = object
    # interfaces.Task
    iface = sys.modules["rai_bench.manipulation_o3de.interfaces"]
    class Task:  # noqa: D401
        pass
    iface.Task = Task
    # Entity SceneConfig
    sb = sys.modules["rai_sim.simulation_bridge"]
    class Entity: pass
    class SceneConfig: pass
    sb.Entity = Entity
    sb.SceneConfig = SceneConfig
    # results tracking light
    rt = sys.modules.setdefault("rai_bench.results_tracking", types.ModuleType("rai_bench.results_tracking"))
    class TaskResult: pass
    rt.TaskResult = TaskResult
    # utils
    utils = sys.modules["rai_bench.utils"]
    utils.define_benchmark_logger = lambda *a, **k: None
    spec = importlib.util.spec_from_file_location("rai_bench_o3de_benchmark_under_test", BENCH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        # fallback: exec stripped Scenario-only via ast
        tree = ast.parse(src)
        scenario_node = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "Scenario":
                scenario_node = node
                break
        if scenario_node is None:
            raise
        module = ast.Module(body=[scenario_node], type_ignores=[])
        ast.fix_missing_locations(module)
        ns = {}
        exec(compile(module, str(BENCH), "exec"), ns)
        return ns["Scenario"]
        raise e
    return mod.Scenario


def test_scenario_rejects_empty_scene_config_path() -> None:
    Scenario = _load_scenario_class()
    task = MagicMock()
    scene = MagicMock()
    with pytest.raises(ValueError, match="scene_config_path"):
        Scenario(task=task, scene_config=scene, scene_config_path="")
    with pytest.raises(ValueError, match="scene_config_path"):
        Scenario(task=task, scene_config=scene, scene_config_path="   ")


def test_scenario_accepts_non_empty_path() -> None:
    Scenario = _load_scenario_class()
    task = MagicMock()
    scene = MagicMock()
    s = Scenario(task=task, scene_config=scene, scene_config_path="configs/1a.yaml")
    assert s.scene_config_path == "configs/1a.yaml"
    assert s.level == "not_declared"
