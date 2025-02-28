from rai_bench.benchmark_model import EntitiesMismatchException, Task  # type: ignore
from rai_sim.o3de.o3de_bridge import SimulationBridge  # type: ignore
from rai_sim.simulation_bridge import SimulationConfig, SpawnedEntity  # type: ignore
import logging
from typing import Union

from rclpy.impl.rcutils_logger import RcutilsLogger

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
