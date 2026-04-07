#!/usr/bin/env python3
"""
Simplified minimal reproduction for ROS2 Humble segfault issue #759.

This is a more focused version that isolates the exact problematic pattern:
- Multiple init/shutdown cycles
- Action servers with multi-threaded executors
- Calling get_action_names_and_types during cleanup

Run with: python3 minimal_repro_simple.py
"""

import threading
import time

import rclpy
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class SimpleActionServer(Node):
    def __init__(self, name):
        super().__init__(name)
        self.server = ActionServer(
            self, NavigateToPose, f"/{name}_action", self.execute
        )

    def execute(self, goal_handle):
        time.sleep(0.01)
        goal_handle.succeed()
        return NavigateToPose.Result()


def test_cycle(i):
    """Single test cycle mimicking the failing test."""
    print(f"Cycle {i}: init")
    rclpy.init()
    
    # Create nodes
    query_node = rclpy.create_node(f"query_{i}")
    server1 = SimpleActionServer(f"server1_{i}")
    server2 = SimpleActionServer(f"server2_{i}")
    
    # Start executors in threads
    executors = []
    threads = []
    for server in [server1, server2]:
        executor = MultiThreadedExecutor()
        executor.add_node(server)
        executors.append(executor)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        threads.append(thread)
    
    time.sleep(0.2)  # Let servers start
    
    # This call can trigger the segfault
    print(f"Cycle {i}: get_action_names_and_types")
    actions = rclpy.action.get_action_names_and_types(query_node)
    print(f"Cycle {i}: found {len(actions)} actions")
    
    # Cleanup - race condition happens here
    print(f"Cycle {i}: cleanup")
    for executor in executors:
        executor.shutdown()
    for thread in threads:
        thread.join(timeout=1.0)
    
    server1.destroy_node()
    server2.destroy_node()
    query_node.destroy_node()
    
    rclpy.shutdown()
    print(f"Cycle {i}: complete\n")


if __name__ == "__main__":
    print("ROS2 Humble Segfault Reproduction - Simplified")
    print("=" * 50)
    
    # Run 10 cycles - segfault typically occurs after a few iterations
    for i in range(1, 11):
        try:
            test_cycle(i)
            time.sleep(0.05)
        except Exception as e:
            print(f"ERROR in cycle {i}: {e}")
            raise
    
    print("=" * 50)
    print("All cycles completed successfully")
