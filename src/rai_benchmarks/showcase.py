import rclpy
from manager import ScenarioManager
from rclpy.executors import MultiThreadedExecutor
from scenarios.longest_object import LongestObjectAuto
from scenarios.move_to_the_left import MoveToTheLeftAuto
from scenarios.place_on_top import PlaceOnTopAuto
from scenarios.replace_types import ReplaceTypesAuto


def main(args=None):
    rclpy.init(args=args)

    manager = ScenarioManager(
        [PlaceOnTopAuto, LongestObjectAuto, MoveToTheLeftAuto, ReplaceTypesAuto],
        list(range(4)),
    )

    executor = MultiThreadedExecutor(2)
    executor.add_node(manager)
    executor.spin()

    manager.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
