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


"""`RunPolicyTool`: a learned policy (VLA) as one long-running, cancellable ACTUATE tool.

Each control tick publishes one action of the current chunk; a new chunk is asked
for when the chunk runs out (or earlier, with prefetch). Observations come from
ROS topics, actions go out through the connector, so egress hooks (EStop,
ConfirmActuation, RateLimit) see every command. On success, timeout, cancel or
error the tool publishes a safe stop (zero twist, JointTrajectoryController hold)
before it returns.
"""

import asyncio
import contextlib
import inspect
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Literal, Optional, Union

import cv2
import numpy as np
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Twist, TwistStamped
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field, model_validator
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rai.communication.ros2 import views
from rai.communication.ros2.egress import MOTION, EgressDenied, type_name
from rai.tools.ros2.base import BaseROS2Tool

from .client import ActionChunk, PolicyClient, PolicyError

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ inputs


class ImageInput(BaseModel):
    """A camera topic as an RGB uint8 (H, W, 3) array (or (3, H, W))."""

    kind: Literal["image"] = "image"
    key: str
    topic: str
    msg_type: str = "sensor_msgs/msg/Image"  # or sensor_msgs/msg/CompressedImage
    size: Optional[tuple[int, int]] = None  # (height, width) sent to the policy
    resize: Literal["pad", "stretch"] = "pad"  # pad keeps the aspect ratio
    channels_first: bool = False


class JointStateInput(BaseModel):
    """A sensor_msgs/JointState field for the given joints, in that order."""

    kind: Literal["joint_state"] = "joint_state"
    key: str
    topic: str
    joints: Optional[list[str]] = None  # None: every joint, in message order
    field: Literal["position", "velocity", "effort"] = "position"


class ArrayInput(BaseModel):
    """The `data` of a std_msgs Float64MultiArray-like message as a vector."""

    kind: Literal["array"] = "array"
    key: str
    topic: str
    msg_type: str = "std_msgs/msg/Float64MultiArray"


PolicyInput = Annotated[
    Union[ImageInput, JointStateInput, ArrayInput], Field(discriminator="kind")
]

# ------------------------------------------------------------------ outputs


class JointTrajectoryOutput(BaseModel):
    """Action dims to a one-point JointTrajectory (arms and grippers on a
    JointTrajectoryController). The safe stop is an empty trajectory, which makes
    the controller hold its current position."""

    kind: Literal["joint_trajectory"] = "joint_trajectory"
    topic: str
    joints: list[str]
    dims: list[int]  # action indices, one per joint
    interface: Literal["position", "velocity"] = "position"
    time_from_start_s: Optional[float] = None  # None: one control period
    clip: Optional[tuple[float, float]] = None

    @model_validator(mode="after")
    def _check(self) -> "JointTrajectoryOutput":
        if len(self.joints) != len(self.dims):
            raise ValueError("joints and dims must have the same length")
        return self

    @property
    def msg_type(self) -> str:
        return type_name(JointTrajectory)

    def build(self, action: np.ndarray, period_s: float) -> Any:
        values = _clip(action[self.dims], self.clip).tolist()
        seconds = self.time_from_start_s or period_s
        point = JointTrajectoryPoint(
            time_from_start=Duration(sec=int(seconds), nanosec=int((seconds % 1) * 1e9))
        )
        setattr(point, f"{self.interface}s", values)
        return JointTrajectory(joint_names=self.joints, points=[point])

    def stop(self) -> Any:
        return JointTrajectory(joint_names=self.joints)


class TwistOutput(BaseModel):
    """Action dims to a Twist (linear x, y, z, angular x, y, z; None is 0). The
    safe stop is a zero twist."""

    kind: Literal["twist"] = "twist"
    topic: str
    dims: list[Optional[int]] = Field(min_length=6, max_length=6)
    stamped: bool = False
    frame_id: str = ""
    clip: Optional[tuple[float, float]] = None

    @property
    def msg_type(self) -> str:
        return type_name(TwistStamped if self.stamped else Twist)

    def _wrap(self, twist: Twist) -> Any:
        if not self.stamped:
            return twist
        msg = TwistStamped(twist=twist)
        msg.header.frame_id = self.frame_id
        return msg

    def build(self, action: np.ndarray, period_s: float) -> Any:
        values = [0.0 if d is None else float(action[d]) for d in self.dims]
        values = _clip(np.asarray(values), self.clip).tolist()
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = values[:3]
        twist.angular.x, twist.angular.y, twist.angular.z = values[3:]
        return self._wrap(twist)

    def stop(self) -> Any:
        return self._wrap(Twist())


PolicyOutput = Annotated[
    Union[JointTrajectoryOutput, TwistOutput], Field(discriminator="kind")
]


def _clip(values: np.ndarray, bounds: Optional[tuple[float, float]]) -> np.ndarray:
    return values if bounds is None else np.clip(values, *bounds)


# ------------------------------------------------------------------ the tool


@dataclass(frozen=True)
class PolicyProgress:
    instruction: str
    elapsed_s: float
    actions: int  # actions published
    queries: int  # policy calls answered
    last_infer_ms: float


class RunPolicyInput(BaseModel):
    instruction: str = Field(
        description="What the robot should do, in the words the policy was trained on"
    )
    max_duration_s: Optional[float] = Field(
        default=None,
        gt=0,
        description="Stop after this many seconds (capped by the skill's limit)",
    )


class RunPolicyTool(BaseROS2Tool):
    """Run a learned visuomotor policy (VLA) until it succeeds, times out or is
    cancelled. See the module docstring and docs/tutorials/policies.md."""

    name: str = "run_policy"
    description: str = (
        "Run the robot's learned motor skill policy with a natural-language "
        "instruction (e.g. 'pick up the red cup'). Blocks until the skill "
        "succeeds, times out or is stopped, then reports what happened."
    )
    args_schema: Any = RunPolicyInput

    policy: Any = Field(exclude=True)  # a PolicyClient
    inputs: list[PolicyInput]
    outputs: list[PolicyOutput] = Field(min_length=1)
    prompt_key: Optional[str] = "prompt"
    extra_observation: dict[str, Any] = Field(default_factory=dict)
    rate_hz: float = Field(default=10.0, gt=0)
    execute_steps: Optional[int] = Field(default=None, gt=0)  # None: whole chunk
    prefetch_steps: int = Field(default=0, ge=0)
    max_duration_s: float = Field(default=60.0, gt=0)
    observation_max_age_s: float = Field(default=0.5, gt=0)
    observation_timeout_s: float = Field(default=2.0, gt=0)
    success_topic: Optional[str] = None  # std_msgs/Bool; True ends the run
    reset_on_start: bool = True
    motion_types: frozenset[str] = MOTION
    progress_period_s: float = Field(default=1.0, gt=0)
    on_progress: Optional[Callable[[PolicyProgress], Any]] = Field(
        default=None, exclude=True
    )

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if not isinstance(self.policy, PolicyClient):
            raise TypeError(f"policy must be a PolicyClient, not {type(self.policy)}")
        for output in self.outputs:
            if output.msg_type not in self.motion_types:
                raise ValueError(
                    f"{output.msg_type} on {output.topic} is not in the MOTION set, so "
                    "e-stop hooks registered with types=MOTION would not see it"
                )
        images = [i.key for i in self.inputs if isinstance(i, ImageInput)]
        keys = [i.key for i in self.inputs]
        if len(set(images)) != len(images) or any(keys.count(k) > 1 for k in images):
            raise ValueError("An image input key must be unique")
        if not self.targets:
            self.metadata["targets"] = tuple(  # type: ignore[index]
                dict.fromkeys(self.connector.resolve(o.topic) for o in self.outputs)
            )

    # -------------------------------------------------------- observations

    async def _read(self, source: Any) -> np.ndarray:
        msg = await self.connector.receive(
            source.topic,
            msg_type=getattr(source, "msg_type", "sensor_msgs/msg/JointState"),
            timeout=self.observation_timeout_s,
            max_age=self.observation_max_age_s,
        )
        if isinstance(source, ImageInput):
            return _rgb(msg, source)
        if isinstance(source, JointStateInput):
            values = getattr(msg, source.field)
            if source.joints is None:
                return np.asarray(values, dtype=np.float64)
            index = {name: i for i, name in enumerate(msg.name)}
            missing = [j for j in source.joints if j not in index]
            if missing:
                raise PolicyError(f"{source.topic} has no joints {missing}")
            if len(values) < len(msg.name):
                raise PolicyError(f"{source.topic} has no {source.field} values")
            return np.asarray([values[index[j]] for j in source.joints], np.float64)
        return np.asarray(msg.data, dtype=np.float64)

    async def observe(self, instruction: str) -> dict[str, Any]:
        """One observation: every input read concurrently (none older than
        observation_max_age_s); vectors that share a key are concatenated."""
        values = await asyncio.gather(*(self._read(s) for s in self.inputs))
        grouped: dict[str, list[np.ndarray]] = {}
        for source, value in zip(self.inputs, values):
            grouped.setdefault(source.key, []).append(value)
        observation: dict[str, Any] = dict(self.extra_observation)
        for key, parts in grouped.items():
            observation[key] = parts[0] if len(parts) == 1 else np.concatenate(parts)
        if self.prompt_key:
            observation[self.prompt_key] = instruction
        return observation

    # -------------------------------------------------------------- actions

    async def _publish(self, action: np.ndarray, period_s: float) -> None:
        for output in self.outputs:
            await self.connector.publish(output.topic, output.build(action, period_s))

    async def _safe_stop(self) -> None:
        """Stop every output; failures are logged, never raised (an e-stop that
        denies the stop command already holds the robot)."""
        for output in self.outputs:
            try:
                async with asyncio.timeout(1.0):
                    await self.connector.publish(output.topic, output.stop())
            except EgressDenied as e:
                logger.warning("Safe stop on %s denied: %s", output.topic, e.reason)
            except Exception:
                logger.exception("Safe stop on %s failed", output.topic)

    async def _query(self, instruction: str, stats: dict[str, Any]) -> ActionChunk:
        observation = await self.observe(instruction)
        start = time.monotonic()
        chunk = await self.policy.infer(observation)
        stats["last_infer_ms"] = (time.monotonic() - start) * 1e3
        stats["queries"] += 1
        width = chunk.shape[1]
        for output in self.outputs:
            dims = [d for d in output.dims if d is not None]
            if dims and max(dims) >= width:
                raise PolicyError(
                    f"The policy returned {width} action dims; {output.topic} "
                    f"uses dim {max(dims)}"
                )
        return chunk[: self.execute_steps] if self.execute_steps else chunk

    async def _progress(self, instruction: str, start: float, stats: dict) -> None:
        progress = PolicyProgress(
            instruction,
            round(time.monotonic() - start, 3),
            stats["actions"],
            stats["queries"],
            round(stats["last_infer_ms"], 1),
        )
        logger.info("Policy progress: %s", progress)
        if self.on_progress is not None:
            try:
                result = self.on_progress(progress)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("on_progress failed")
        try:  # surfaces in astream_events of the agent run, if there is one
            from langchain_core.callbacks.manager import adispatch_custom_event

            await adispatch_custom_event("rai_policy_progress", asdict(progress))
        except Exception:  # not inside a LangChain run
            pass

    # ----------------------------------------------------------------- run

    async def _acall(
        self, instruction: str, max_duration_s: Optional[float] = None
    ) -> str:
        for output in self.outputs:
            self.check_writable(output.topic)
        for source in self.inputs:
            self.check_readable(source.topic)
        duration = min(max_duration_s or self.max_duration_s, self.max_duration_s)
        succeeded = threading.Event()
        subscription = None
        if self.success_topic:
            subscription = self.connector.subscribe(
                self.success_topic,
                lambda msg: succeeded.set() if msg.data else None,
                msg_type="std_msgs/msg/Bool",
            )
        stats: dict[str, Any] = {"actions": 0, "queries": 0, "last_infer_ms": 0.0}
        start = time.monotonic()
        pending: Optional[asyncio.Task[ActionChunk]] = None
        deadline = asyncio.timeout(duration)
        try:
            if self.reset_on_start:
                await self.policy.reset()
            async with deadline:
                pending = await self._control(instruction, start, stats, succeeded)
            outcome = "succeeded"
        except asyncio.CancelledError:
            logger.info("Policy run cancelled; sending the safe stop")
            await asyncio.shield(self._safe_stop())
            raise
        except BaseException as e:
            if isinstance(e, TimeoutError) and deadline.expired():
                outcome = f"ran for its {duration:g} s limit"
                if self.success_topic:
                    outcome = (
                        f"stopped at its {duration:g} s limit with no success signal"
                    )
            else:
                await asyncio.shield(self._safe_stop())
                if not isinstance(e, Exception):
                    raise
                elapsed = time.monotonic() - start
                what = e if isinstance(e, PolicyError) else f"{type(e).__name__}: {e}"
                raise ToolException(
                    f"Policy stopped after {elapsed:.1f} s and {stats['actions']} "
                    f"actions, robot halted. {what}"
                ) from e
        finally:
            if pending is not None:
                pending.cancel()
            if subscription is not None:
                subscription.close()
        await asyncio.shield(self._safe_stop())
        return (
            f"Policy {outcome} after {time.monotonic() - start:.1f} s: "
            f"{stats['actions']} actions from {stats['queries']} policy calls; "
            "robot halted."
        )

    async def _control(
        self,
        instruction: str,
        start: float,
        stats: dict[str, Any],
        succeeded: threading.Event,
    ) -> Optional["asyncio.Task[ActionChunk]"]:
        """The control loop; returns when the success signal arrives."""
        loop = asyncio.get_running_loop()
        period = 1.0 / self.rate_hz
        queue: deque[np.ndarray] = deque()
        pending: Optional[asyncio.Task[ActionChunk]] = None
        next_tick = loop.time()
        next_progress = start + self.progress_period_s
        try:
            while not succeeded.is_set():
                if not queue:
                    if pending is None:
                        pending = asyncio.ensure_future(self._query(instruction, stats))
                    task, pending = pending, None
                    queue.extend(await task)
                    next_tick = max(next_tick, loop.time())  # no burst after a wait
                    continue  # success may have arrived during inference
                if pending is None and len(queue) <= self.prefetch_steps:
                    pending = asyncio.ensure_future(self._query(instruction, stats))
                await self._publish(queue.popleft(), period)
                stats["actions"] += 1
                if time.monotonic() >= next_progress:
                    next_progress += self.progress_period_s
                    await self._progress(instruction, start, stats)
                next_tick += period
                await asyncio.sleep(max(0.0, next_tick - loop.time()))
        except BaseException:
            if pending is not None:
                pending.cancel()
                with contextlib.suppress(BaseException):
                    await pending
            raise
        return pending


def _rgb(msg: Any, source: ImageInput) -> np.ndarray:
    pixels = views.image(msg)
    encoding = "bgr8" if not hasattr(msg, "encoding") else msg.encoding
    if pixels.ndim == 3 and encoding.startswith("bgr"):
        pixels = pixels[..., 2::-1]
    elif pixels.ndim == 3 and pixels.shape[2] == 4:
        pixels = pixels[..., :3]
    if source.size is not None:
        pixels = _resize(np.ascontiguousarray(pixels), source.size, source.resize)
    if source.channels_first and pixels.ndim == 3:
        pixels = pixels.transpose(2, 0, 1)
    return np.ascontiguousarray(pixels)


def _resize(pixels: np.ndarray, size: Sequence[int], mode: str) -> np.ndarray:
    height, width = size
    if mode == "stretch":
        return cv2.resize(pixels, (width, height), interpolation=cv2.INTER_AREA)
    scale = min(height / pixels.shape[0], width / pixels.shape[1])
    h, w = (
        max(1, round(pixels.shape[0] * scale)),
        max(1, round(pixels.shape[1] * scale)),
    )
    scaled = cv2.resize(pixels, (w, h), interpolation=cv2.INTER_AREA)
    out = np.zeros((height, width, *pixels.shape[2:]), dtype=pixels.dtype)
    top, left = (height - h) // 2, (width - w) // 2
    out[top : top + h, left : left + w] = scaled
    return out
