# Copyright (C) 2026 Robotec.AI
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

"""ReplayPlayer: recorded topics published into a live ROS graph."""

import logging
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional, Self, Union

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from rai.communication.ros2.egress import MOTION
from rai.communication.ros2.runtime import ROS2Runtime, message_class
from rai.recorder import Recording

from .source import Bag, open_bag

logger = logging.getLogger(__name__)

_RELIABLE = QoSProfile(
    depth=100,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
_LATCHED = QoSProfile(
    depth=100,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class ReplayPlayer:
    """Publishes the ROS topics of a recording with real publishers, at the
    recorded pace (scaled by ``rate``), so the real connector, tools and
    aggregators run unchanged against recorded data.

    With ``clock=True`` it publishes ``/clock`` with the recording's time, so
    nodes on sim time (``ROS2Connector(use_sim_time=True)``) see the recorded
    time in ``now()``. Topics of a motion type (``MOTION``: velocity commands,
    trajectories, ...) are never replayed unless ``include_motion=True``: a
    recording played into a robot's graph must not drive it.

    Publishers are created in the constructor, so consumers can connect
    (:meth:`wait_for_subscribers`) before :meth:`start`::

        with ReplayPlayer("run.mcap", clock=True) as player:
            player.wait_for_subscribers(["/camera/image_raw"])
            player.start()
            player.wait()
    """

    def __init__(
        self,
        source: Union[str, Path, Recording, Bag],
        *,
        topics: Optional[Iterable[str]] = None,
        exclude: Iterable[str] = (),
        rate: float = 1.0,
        loop: bool = False,
        start: float = 0.0,
        clock: bool = False,
        clock_hz: float = 100.0,
        remap: Optional[Mapping[str, str]] = None,
        latched: Iterable[str] = ("/tf_static",),
        include_motion: bool = False,
        node_name: Optional[str] = None,
    ) -> None:
        """
        Args:
            source: a flight recording, a rosbag2 file or directory, or a Bag.
            topics: recorded topics to play (default: all ROS topics).
            exclude: recorded topics not to play.
            rate: time scale; 2.0 plays twice as fast.
            loop: start over at the end (``/clock`` then jumps back).
            start: seconds into the recording to start at.
            clock: publish ``/clock`` (the recording's ``/clock`` is not played).
            clock_hz: how often ``/clock`` ticks between messages.
            remap: recorded name -> published name.
            latched: topics published with transient-local durability.
            include_motion: also play topics of a ``MOTION`` type.
            node_name: the player's node (default ``rai_replay_<id>``).
        """
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.bag = open_bag(source)
        self.rate = rate
        self.loop = loop
        self.clock = clock
        self.clock_hz = clock_hz
        self.published = 0
        self.loops = 0
        self.stamp = 0  # recording time (ns) of the last message played
        self._remap = dict(remap or {})
        excluded = set(exclude) | ({"/clock"} if clock else set())
        selected = set(self.bag.topics) if topics is None else set(topics)
        unknown = selected - set(self.bag.topics)
        if unknown:
            raise ValueError(f"Not in the recording: {', '.join(sorted(unknown))}")
        self.topics: dict[str, str] = {}
        for name in sorted(selected - excluded):
            type_ = self.bag.topics[name]
            if type_ in MOTION and not include_motion:
                logger.warning(
                    "ReplayPlayer: not replaying %s (%s moves hardware;"
                    " include_motion=True plays it)",
                    name,
                    type_,
                )
                continue
            self.topics[name] = type_
        self.skipped = sorted(selected - excluded - set(self.topics))
        self.first, self.last = self.bag.bounds()
        self._from = self.first + int(start * 1e9)
        self._seek: Optional[int] = None
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._finished = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._runtime = ROS2Runtime(node_name or f"rai_replay_{uuid.uuid4().hex[:8]}")
        node = self._runtime.node
        latched = set(latched)
        try:
            self._publishers = {
                name: node.create_publisher(
                    message_class(type_),
                    self._remap.get(name, name),
                    _LATCHED if name in latched else _RELIABLE,
                )
                for name, type_ in self.topics.items()
            }
            self._clock_pub: Any = None
            if clock:
                from rosgraph_msgs.msg import Clock

                self._clock_msg = Clock
                self._clock_pub = node.create_publisher(Clock, "/clock", _RELIABLE)
        except BaseException:
            self._runtime.close()
            raise

    # ------------------------------------------------------------ control

    def start(self) -> Self:
        """Start playing (in a background thread)."""
        if self._thread is not None:
            raise RuntimeError("ReplayPlayer already started")
        self._thread = threading.Thread(
            target=self._play, name="rai_replay_player", daemon=True
        )
        self._thread.start()
        return self

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def seek(self, seconds: float) -> None:
        """Continue playing from seconds into the recording."""
        self._seek = self.first + int(seconds * 1e9)
        self._finished.clear()

    @property
    def done(self) -> bool:
        return self._finished.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until the recording has played to its end (never, with loop)."""
        return self._finished.wait(timeout)

    def wait_for_subscribers(
        self, topics: Optional[Iterable[str]] = None, timeout: float = 10.0
    ) -> bool:
        """Wait until every topic (recorded names; default all played) has a
        subscriber. Returns False on timeout."""
        names = list(self.topics) if topics is None else list(topics)
        deadline = time.monotonic() + timeout
        while True:
            if all(self._publishers[n].get_subscription_count() > 0 for n in names):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)

    def close(self) -> None:
        """Stop playing and destroy the publishers. Idempotent."""
        self._stop.set()
        self._resume.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(5.0)
        self._runtime.close()

    stop = close

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ playing

    def _play(self) -> None:
        try:
            while not self._stop.is_set():
                start, self._seek = (
                    (self._seek, None) if self._seek is not None else (self._from, None)
                )
                if self._play_from(start):
                    continue  # sought
                self.loops += 1
                if not self.loop:
                    self._finished.set()
                    while not self._stop.is_set() and self._seek is None:
                        self._stop.wait(0.05)
        except Exception:
            if not self._stop.is_set() and not self._runtime.closed:
                logger.exception("ReplayPlayer failed")
            self._finished.set()

    def _play_from(self, start: int) -> bool:
        """Play from start to the end; True if interrupted by a seek."""
        t0, wall0 = start, time.monotonic()
        for message in self.bag.messages(self.topics, start=start):
            target = wall0 + (message.stamp - t0) / 1e9 / self.rate
            while True:
                if self._stop.is_set():
                    return False
                if self._seek is not None:
                    return True
                if not self._resume.is_set():
                    paused = time.monotonic()
                    self._resume.wait()
                    wall0 += time.monotonic() - paused
                    target += time.monotonic() - paused
                    continue
                now = time.monotonic()
                if now >= target:
                    break
                if self._clock_pub is not None:
                    self._tick(t0 + int((now - wall0) * self.rate * 1e9))
                    time.sleep(min(target - now, 1.0 / self.clock_hz))
                else:
                    time.sleep(min(target - now, 0.05))
            if self._clock_pub is not None:
                self._tick(message.stamp)
            self._publishers[message.topic].publish(message.data)
            self.stamp = message.stamp
            self.published += 1
        return False

    def _tick(self, stamp: int) -> None:
        msg = self._clock_msg()
        msg.clock.sec, msg.clock.nanosec = divmod(stamp, 1_000_000_000)
        self._clock_pub.publish(msg)
