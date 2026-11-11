# Learned-policy skills (VLA)

A vision-language-action policy (pi0, pi0.5, GR00T N1.x, SmolVLA, ACT) is a good motor skill and a poor planner. RAI wraps one as a single long-running tool, `run_policy(instruction, max_duration_s)`, so the LLM agent decides _when_ to pick up the cup and the policy decides _how_.

`RunPolicyTool` reads observations from ROS topics, asks a remote policy server for an action chunk, publishes one action per control tick, and stops on a success signal, its time limit, cancellation or an error. Every exit publishes a safe stop: a zero twist, and an empty `JointTrajectory`, which makes a JointTrajectoryController hold its position. Commands go through the connector, so egress hooks (e-stop, confirmation, rate limits) see every one of them: an e-stop denial ends the run and the LLM gets an error tool message.

Install the clients with the `policies` extra:

```bash
pip install 'rai_core[policies]'
```

## Policy servers

| Client                                            | Server                                                   | Protocol             |
| ------------------------------------------------- | -------------------------------------------------------- | -------------------- |
| `OpenPIClient("ws://gpu-box:8000")`               | openpi `scripts/serve_policy.py` (pi0, pi0-FAST, pi0.5)  | websocket, msgpack   |
| `Gr00tClient("gpu-box", 5555, action_keys=[...])` | Isaac GR00T `gr00t/eval/run_gr00t_server.py`             | ZMQ REQ/REP, msgpack |
| `HttpPolicyClient("http://gpu-box:8000/infer")`   | any server that takes and returns msgpack over HTTP POST | HTTP, msgpack        |

All of them implement `PolicyClient` (`async infer(observation) -> (T, D) array`, `reset()`, `close()`), time out, and reconnect after an error or a cancelled call. GR00T observation keys are `video.<key>`, `state.<key>` and `language.<key>` with the checkpoint's modality keys; `await client.modality_config()` lists them.

LeRobot policies (ACT, SmolVLA, pi0 ports) can be served by wrapping `policy.select_action` in a small HTTP endpoint for `HttpPolicyClient`. LeRobot's own `remote_inference` protocol (Zenoh) is not supported yet.

## Configuring a skill

```python
from rai.communication.ros2 import ROS2Connector
from rai.tools.policies import OpenPIClient, RunPolicyTool

pick = RunPolicyTool(
    connector=ROS2Connector(),
    policy=OpenPIClient("ws://gpu-box:8000", timeout_s=2.0),
    inputs=[
        {"kind": "image", "key": "observation/image", "topic": "/camera/color/image_raw", "size": (224, 224)},
        {"kind": "image", "key": "observation/wrist_image", "topic": "/wrist/image_raw", "size": (224, 224)},
        {"kind": "joint_state", "key": "observation/state", "topic": "/joint_states",
         "joints": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]},
    ],
    outputs=[
        {"kind": "joint_trajectory", "topic": "/arm_controller/joint_trajectory",
         "joints": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
         "dims": [0, 1, 2, 3, 4, 5, 6]},
        {"kind": "joint_trajectory", "topic": "/gripper_controller/joint_trajectory",
         "joints": ["finger_joint"], "dims": [7], "clip": (0.0, 0.04)},
    ],
    rate_hz=15.0,
    execute_steps=5,       # run 5 actions of each chunk, then ask again
    prefetch_steps=2,      # ask 2 actions early to hide inference latency
    max_duration_s=60.0,   # the most the LLM can ask for
    success_topic="/task_done",  # optional std_msgs/Bool
)
```

-   **Inputs.** `image` (Image or CompressedImage, sent as RGB uint8, optionally resized with padding or stretched, `channels_first` for CHW), `joint_state` (one field of the listed joints, in that order) and `array` (`std_msgs/Float64MultiArray`). Vector inputs that share a key are concatenated in order. The instruction goes under `prompt_key` (default `"prompt"`); constant entries go in `extra_observation`. Each step reads the latest message of every input and fails if one is older than `observation_max_age_s`, so a dead camera stops the robot instead of feeding the policy an old frame.
-   **Outputs.** `joint_trajectory` (position or velocity, for arms and grippers) and `twist` (six action indices for linear and angular x, y, z, `None` for 0; `stamped=True` for TwistStamped). `clip` bounds the values. Output types must be in the MOTION set, so e-stop hooks registered with `types=MOTION` always see them.
-   **Progress.** Every `progress_period_s` the tool calls `on_progress(PolicyProgress)` and emits a `rai_policy_progress` custom event into the agent's LangChain event stream.
-   **Stopping.** Cancelling the tool (the agent's stop or interrupt) stops publishing at once and sends the safe stop. A dry run against `FakePolicyServer` (scripted chunks, openpi protocol) checks the ROS wiring without a GPU.

See the [API reference](../api/policies.md) for every parameter.
