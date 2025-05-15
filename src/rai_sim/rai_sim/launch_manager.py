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

import asyncio
import multiprocessing
from multiprocessing.synchronize import Event
from typing import Optional

from launch import LaunchDescription, LaunchService


class ROS2LaunchManager:
    def __init__(self) -> None:
        self._stop_event: Optional[Event] = None
        self._process: Optional[multiprocessing.Process] = None

    def start(self, launch_description: LaunchDescription) -> None:
        self._stop_event = multiprocessing.Event()
        self._process = multiprocessing.Process(
            target=self._run_process,
            args=(self._stop_event, launch_description),
            daemon=True,
        )
        self._process.start()

    def shutdown(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        if self._process:
            self._process.join()

    def _run_process(
        self, stop_event: Event, launch_description: LaunchDescription
    ) -> None:
        loop = asyncio.get_event_loop()
        asyncio.set_event_loop(loop)
        launch_service = LaunchService()
        launch_service.include_launch_description(launch_description)
        # launch description launched
        launch_task = loop.create_task(launch_service.run_async())
        # when stop event set
        loop.run_until_complete(loop.run_in_executor(None, stop_event.wait))
        if not launch_task.done():
            # XXX (jmatejcz) the shutdown function sends shutdown signal to all
            # nodes launch with launch description which should do the trick
            # but some nodes are stubborn and there is a possibility
            # that they don't close. If this will happen sending PKILL for all
            # ros nodes will be needed
            shutdown_task = loop.create_task(
                launch_service.shutdown(),
            )
            # shutdown task should complete when all nodes are closed
            # but wait also for launch task to close just to be sure
            loop.run_until_complete(asyncio.gather(shutdown_task, launch_task))
