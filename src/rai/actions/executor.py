import logging
import threading
from typing import TYPE_CHECKING, Callable, List, Optional

from .actions import Action

if TYPE_CHECKING:
    from rai.message import Message
    from rai.scenario_engine.scenario_engine import ScenarioRunner


class Executor:
    def __init__(self, action: Action, logging_level: int = logging.INFO):
        self.action = action
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging_level)

    def __call__(self, runner: "ScenarioRunner") -> Optional[threading.Thread]:
        self.logger.info(f"Executing action: {self.action.__class__.__name__}")
        if self.action.separate_thread:
            thread = threading.Thread(target=self.action.run, args=(runner,))
            thread.start()
            return thread
        else:
            self.action.run(runner)
            return None


class ConditionalExecutor(Executor):
    def __init__(
        self,
        action: Action,
        condition: Callable[[List["Message"]], bool],
        logging_level: int = logging.INFO,
    ):
        super().__init__(action, logging_level=logging_level)
        self.action = action
        self.condition = condition

    def __call__(self, runner: "ScenarioRunner") -> Optional[threading.Thread]:
        condition_met = self.condition(runner.history)
        if condition_met:
            self.logger.info(f"Condition met.")
            return super().__call__(runner)
        return None
