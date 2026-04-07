#!/usr/bin/env python3
"""
Minimal reproduction script for ROS2 Humble segfault issue #759.

This script reproduces the segmentation fault that occurs when:
1. Multiple rclpy.init()/shutdown() cycles are performed
2. Action servers are created with multi-threaded executors
3. rclpy.action.get_action_names_and_types() is called
4. Resources are cleaned up

The issue appears to be a race condition in ROS2 Humble's C++ layer
when accessing action server information during/after shutdown.
"""

import threading
import time
import uuid

import rclpy
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class TestActionServer(Node):
    """Minimal action server for testing."""

    def __init__(self, action_name: str):
        super().__init__(f"test_action_server_{str(uuid.uuid4())[-12:]}")
        self.action_server = ActionServer(
            self,
            action_type=NavigateToPose,
            action_name=action_name,
            execute_callback=self.handle_action,
            goal_callback=self.goal_accepted,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.cancelled = False

    def handle_action(self, goal_handle: ServerGoalHandle) -> NavigateToPose.Result:
        """Simple action handler that completes quickly."""
        for i in range(1, 11):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = NavigateToPose.Result()
                result.error_code = 3
                return result
            time.sleep(0.01)

        goal_handle.succeed()
        result = NavigateToPose.Result()
        result.error_code = NavigateToPose.Result.NONE
        return result

    def goal_accepted(self, goal_handle: ServerGoalHandle) -> GoalResponse:
        return GoalResponse.ACCEPT

    def cancel_callback(self, cancel_request) -> CancelResponse:
        self.cancelled = True
        return CancelResponse.ACCEPT


def run_test_iteration(iteration: int):
    """Run a single test iteration that mimics the failing test pattern."""
    print(f"\n{'='*60}")
    print(f"Iteration {iteration}")
    print(f"{'='*60}")

    # Initialize rclpy (mimics ros_setup fixture)
    print("Initializing rclpy...")
    rclpy.init()

    try:
        # Create a node for querying actions
        print("Creating query node...")
        query_node = rclpy.create_node(f"query_node_{iteration}")

        # Create two action servers (mimics test setup)
        print("Creating action servers...")
        action_name_1 = f"/test_iteration_{iteration}_action_1"
        action_name_2 = f"/test_iteration_{iteration}_action_2"
        
        server1 = TestActionServer(action_name=action_name_1)
        server2 = TestActionServer(action_name=action_name_2)

        # Spin servers in separate threads with multi-threaded executors
        print("Starting executors in separate threads...")
        executors = []
        threads = []
        
        for server in [server1, server2]:
            executor = MultiThreadedExecutor()
            executor.add_node(server)
            executors.append(executor)
            
            thread = threading.Thread(target=executor.spin, daemon=True)
            thread.start()
            threads.append(thread)

        # Wait for servers to be ready
        print("Waiting for action servers to be ready...")
        time.sleep(0.2)

        # Call get_action_names_and_types - this is where the segfault occurs
        print("Calling rclpy.action.get_action_names_and_types()...")
        try:
            actions = rclpy.action.get_action_names_and_types(query_node)
            print(f"Found {len(actions)} actions:")
            for action_name, action_types in actions:
                print(f"  - {action_name}: {action_types}")
        except Exception as e:
            print(f"ERROR during get_action_names_and_types: {e}")

        # Cleanup - this is critical and where race conditions occur
        print("Shutting down executors...")
        for executor in executors:
            try:
                executor.shutdown()
            except Exception as e:
                print(f"Error shutting down executor: {e}")

        print("Waiting for threads to finish...")
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=2.0)

        print("Destroying nodes...")
        try:
            server1.destroy_node()
        except Exception as e:
            print(f"Error destroying server1: {e}")
        
        try:
            server2.destroy_node()
        except Exception as e:
            print(f"Error destroying server2: {e}")
        
        try:
            query_node.destroy_node()
        except Exception as e:
            print(f"Error destroying query_node: {e}")

    finally:
        # Shutdown rclpy (mimics ros_setup fixture cleanup)
        print("Shutting down rclpy...")
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f"Error during rclpy.shutdown(): {e}")

    print(f"Iteration {iteration} completed successfully")


def main():
    """Run multiple test iterations to reproduce the segfault."""
    print("="*60)
    print("ROS2 Humble Segfault Reproduction Script")
    print("Issue: https://github.com/RobotecAI/rai/issues/759")
    print("="*60)
    print("\nThis script reproduces the segfault by:")
    print("1. Running multiple rclpy.init()/shutdown() cycles")
    print("2. Creating action servers with multi-threaded executors")
    print("3. Calling rclpy.action.get_action_names_and_types()")
    print("4. Cleaning up resources")
    print("\nThe segfault typically occurs after several iterations,")
    print("especially during the 'forbidden' test pattern.")
    print("="*60)

    # Run multiple iterations to trigger the race condition
    # The segfault in CI occurred on the 5th test in the sequence
    num_iterations = 10
    
    for i in range(1, num_iterations + 1):
        try:
            run_test_iteration(i)
            # Small delay between iterations
            time.sleep(0.1)
        except Exception as e:
            print(f"\n{'!'*60}")
            print(f"EXCEPTION in iteration {i}: {e}")
            print(f"{'!'*60}")
            raise

    print("\n" + "="*60)
    print(f"All {num_iterations} iterations completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
