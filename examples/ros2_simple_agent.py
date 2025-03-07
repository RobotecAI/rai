import rclpy
from rai.agents.simple_agent import SimpleAgent
from rai.communication.ros2.connectors import ROS2HRIConnector


def main():
    rclpy.init()
    connector = ROS2HRIConnector(targets=["/to_human"], sources=["/from_human"])
    agent = SimpleAgent(connectors={"ros2": connector})  # type: ignore
    agent.run()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
