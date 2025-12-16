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

import inspect
import socket
from typing import Any, Callable



class ObservabilityMeta(type):
    TRACKED_METHODS: set[str] = {
        "send_message",
        "receive_message",
        "service_call",
        "call_service",
        "create_service",
    }

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ):
        cls = super().__new__(mcls, name, bases, namespace)

        for base in cls.__mro__:
            for attr_name, attr_value in base.__dict__.items():
                if attr_name not in mcls.TRACKED_METHODS:
                    continue

                if not inspect.isfunction(attr_value):
                    continue

                resolved = getattr(cls, attr_name, None)
                if resolved is not attr_value:
                    continue

                wrapped = mcls._wrap(attr_value)
                setattr(cls, attr_name, wrapped)

        return cls

    @staticmethod
    def _wrap(fn: Callable[..., object]) -> Callable[..., object]:
        def wrapped(self, *args: object, **kwargs: object) -> object:
            obs: ObservabilityModule | None = getattr(self, "observability_module", None)
            target_or_source = kwargs.get(
                "target",
                kwargs.get(
                    "source",
                    kwargs.get("service_name", "unknown")
                ),
            )

            def _send_message(message: str) -> None:
                if obs is None:
                    return

                if not obs.registered:
                    obs.registered = True
                    obs.connection.send(f"<{obs.namespace}:{obs.name}> register\n".encode())
                
                try:
                    obs.connection.send(message.encode())
                except Exception as e:
                    print(f"Error sending observability message: {e}")
                    obs.reconnect()

            if obs is None:
                return fn(self, *args, **kwargs)

            _send_message(
                f"<{obs.namespace}:{obs.name}> "
                f"{fn.__name__} {target_or_source} opened\n"
            )

            result = fn(self, *args, **kwargs)

            _send_message(
                f"<{obs.namespace}:{obs.name}> "
                f"{fn.__name__} {target_or_source} closed\n"
            )

            return result

        wrapped.__name__ = fn.__name__
        wrapped.__qualname__ = fn.__qualname__
        wrapped.__doc__ = fn.__doc__

        return wrapped


class ObservabilityModule:
    def __init__(self, name: str, namespace: str, ip: str, port: int) -> None:
        self.name = name
        self.namespace = namespace
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection.connect((ip, port))
        self.ip = ip
        self.port = port
        self.registered = False
        
        # Auto-register on init to ensure node appears in graph even without activity
        try:
            self.connection.send(f"<{self.namespace}:{self.name}> register\n".encode())
            self.registered = True
        except Exception as e:
            print(f"Error registering observability module: {e}")

    def reconnect(self) -> None:
        try:
            self.connection.close()
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.connect((self.ip, self.port))
            # Re-register on reconnect
            self.connection.send(f"<{self.namespace}:{self.name}> register\n".encode())
        except Exception as e:
            print(f"Error reconnecting observability module: {e}")
