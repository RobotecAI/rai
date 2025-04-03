import rclpy
from rai.utils import wait_for_shutdown
from rai_open_set_vision.agents import GroundedSamAgent, GroundingDinoAgent


def main():
    rclpy.init()
    agent1 = GroundingDinoAgent()
    agent2 = GroundedSamAgent()
    agent1.run()
    agent2.run()
    wait_for_shutdown([agent1, agent2])
    rclpy.shutdown()


if __name__ == "__main__":
    main()
