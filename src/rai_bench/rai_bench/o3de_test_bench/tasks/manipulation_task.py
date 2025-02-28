import logging
from typing import Union

from rclpy.impl.rcutils_logger import RcutilsLogger

from rai_bench.benchmark_model import Task  # type: ignore

loggers_type = Union[RcutilsLogger, logging.Logger]


class ManipulationTask(Task):
    def __init__(self, logger: loggers_type | None = None):
        super().__init__(logger)
        self.initially_misplaced_now_correct = 0
        self.initially_misplaced_still_incorrect = 0
        self.initially_correct_still_correct = 0
        self.initially_correct_now_incorrect = 0

    def reset_values(self):
        self.initially_misplaced_now_correct = 0
        self.initially_misplaced_still_incorrect = 0
        self.initially_correct_still_correct = 0
        self.initially_correct_now_incorrect = 0
