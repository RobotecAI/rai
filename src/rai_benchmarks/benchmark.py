import rclpy
from manager import RaiBenchmarkManager
from rclpy.executors import MultiThreadedExecutor
from scenarios.longest_object import LongestObject
from scenarios.move_to_the_left import MoveToTheLeft
from scenarios.place_on_top import PlaceOnTop
from scenarios.replace_types import ReplaceTypes


def main(args=None):
    rclpy.init(args=args)

    manager = RaiBenchmarkManager(
        [PlaceOnTop, LongestObject, MoveToTheLeft, ReplaceTypes], list(range(4))
    )

    executor = MultiThreadedExecutor(2)
    executor.add_node(manager)
    executor.spin()

    manager.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
