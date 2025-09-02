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

import signal
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class TimeoutError(Exception):
    """Raised when an operation times out."""

    pass


def timeout(seconds: float, timeout_message: str = None) -> Callable[[F], F]:
    """
    Decorator that adds timeout functionality to a function.

    Parameters
    ----------
    seconds : float
        Timeout duration in seconds
    timeout_message : str, optional
        Custom timeout message. If not provided, a default message will be used.

    Returns
    -------
    Callable
        Decorated function with timeout functionality

    Raises
    ------
    TimeoutError
        When the decorated function exceeds the specified timeout

    Examples
    --------
    >>> @timeout(5.0, "Operation timed out")
    ... def slow_operation():
    ...     import time
    ...     time.sleep(10)
    ...     return "Done"
    >>>
    >>> try:
    ...     result = slow_operation()
    ... except TimeoutError as e:
    ...     print(f"Timeout: {e}")
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                message = (
                    timeout_message
                    or f"Function '{func.__name__}' timed out after {seconds} seconds"
                )
                raise TimeoutError(message)

            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))

            try:
                return func(*args, **kwargs)
            finally:
                # Clean up timeout
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator


def timeout_method(seconds: float, timeout_message: str = None) -> Callable[[F], F]:
    """
    Decorator that adds timeout functionality to a method.
    Similar to timeout but designed for class methods.

    Parameters
    ----------
    seconds : float
        Timeout duration in seconds
    timeout_message : str, optional
        Custom timeout message. If not provided, a default message will be used.

    Returns
    -------
    Callable
        Decorated method with timeout functionality

    Examples
    --------
    >>> class MyClass:
    ...     @timeout_method(3.0, "Method timed out")
    ...     def slow_method(self):
    ...         import time
    ...         time.sleep(5)
    ...         return "Done"
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            def timeout_handler(signum, frame):
                message = (
                    timeout_message
                    or f"Method '{func.__name__}' of {self.__class__.__name__} timed out after {seconds} seconds"
                )
                raise TimeoutError(message)

            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))

            try:
                return func(self, *args, **kwargs)
            finally:
                # Clean up timeout
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator
