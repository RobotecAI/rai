# Turtlebot4 tutorial

The following tutorial will help you get started with [rai](https://github.com/RobotecAI/rai)
using Turtlebot4 simulation. The step by step video tutorial is available [here](https://robotecai-my.sharepoint.com/:v:/g/personal/bartlomiej_boczek_robotec_ai/EfPTiZscCTROtmtoyHv_ykIBFKN5qh1pecxfLmLI6I4QeA?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=0QJnGk).

## Step by step instructions

0. Install nav2 packages

   ```shell
      sudo apt install "ros-${ROS_DISTRO}-nav2-*"
   ```

1. Clone and install [rai](https://github.com/RobotecAI/rai)

   1. Follow steps from [1. Setting up the workspace](https://github.com/RobotecAI/rai?tab=readme-ov-file#1-setting-up-the-workspace)
   2. Then follow steps from [2. Build the project](https://github.com/RobotecAI/rai?tab=readme-ov-file#2-build-the-project)

2. O3DE simulation setup

   1. Download turtlebot4 simulation binary matching your platform from :

      - Ubuntu 22.04 & ros2 humble: [link](https://robotec-ml-roscon2024-demos.s3.eu-central-1.amazonaws.com/ROSCON_Release/Turtlebot4Tutorial_1.0.0_jammyhumble.zip)
      - Ubuntu 24.04 & ros2 jazzy: [link](https://robotec-ml-roscon2024-demos.s3.eu-central-1.amazonaws.com/ROSCON_Release/Turtlebot4Tutorial_1.0.0_noblejazzy.zip)

      ```bash
      cd rai
      # ubuntu 22.04 ~ humble
      wget https://robotec-ml-roscon2024-demos.s3.eu-central-1.amazonaws.com/ROSCON_Release/Turtlebot4Tutorial_1.0.0_jammyhumble.zip

      # ubuntu 24.04 ~ jazzy
      wget https://robotec-ml-roscon2024-demos.s3.eu-central-1.amazonaws.com/ROSCON_Release/Turtlebot4Tutorial_1.0.0_noblejazzy.zip
      ```

   2. Unzip it:

      ```bash
      # for Ubuntu 22.04 jammy and ros2 humble
      unzip -d src/examples/turtlebot4/simulation Turtlebot4Tutorial_1.0.0_jammyhumble.zip

      # for Ubuntu 24.04 noble and ros2 jazzy
      unzip -d src/examples/turtlebot4/simulation Turtlebot4Tutorial_1.0.0_noblejazzy.zip
      ```

3. Setup your LLM vendor in [config.toml](../../../config.toml) and configure access keys
   as in [docs/vendors.md](../../../docs/vendors.md). OpenAI or AWS Bedrock are recommended
   for models, since current local `ollama` models don't support vision & tool calling.
   `ollama` and `llama:3.2` as an embedding model generate acceptable quality of robot's
   indentity in this tutorial creating identity

4. Configure `rai_whoami_node` (based on ["Your robot identity in RAI"](https://github.com/RobotecAI/rai/blob/development/docs/create_robots_whoami.md) tutorial):

   1. Create `whoami` package for turtlebot4 in `src/examples/turtlebot4`

      ```bash
      ./scripts/create_rai_ws.sh --name turtlebot4 --destination-directory src/examples
      ```

   > **TIP**  
   > Skip steps 2-4 by downloading generated files [here](https://robotecai-my.sharepoint.com/:u:/g/personal/bartlomiej_boczek_robotec_ai/EbPZSEdXYaRGoeecu6oJg6QBsI4ZOe_mrU3uOtOflnIjQg?e=HX8ZHB) unzip them to `src/examples/turtlebot4_whoami/description/generated` with a command:
   > `unzip -d src/examples/turtlebot4_whoami/description turtlebot4_whoami_generated.zip`

   2. Download official turtlebot4 [data sheet](https://bit.ly/3KCp3Du) into
      `src/examples/turtlebot4_whoami/description/documentation`
   3. Download [image](https://s3.amazonaws.com/assets.clearpathrobotics.com/wp-content/uploads/2022/03/16113604/Turtlebot-4-20220207.44.png) of turtlebot4 into `src/examples/turtlebot4_whoami/description/images`
   4. Create robot's identity. Run the `parse_whoami_package`. This will process the documentation, building
      it into a vector database, which is used by RAI agent to reason about its identity.

      ```bash
      ./scripts/parse_whoami_package.sh src/examples/turtlebot4_whoami
      # you will be asked to press `y` to continue
      ```

   5. Build the workspace which now includes the new package

      ```bash
      deactivate # if poetry env is activated
      colcon build --symlink-install
      ```

   6. Ensure `turtlebot4_whoami` package has been built:

      ```bash
      . ./setup_shell.sh
      ros2 pkg list | grep turtlebot4
      ```

   Congratulations! Your `rai_whoami_node` is configured. In the following steps
   your RAI agent will assume a turtlebot4 personality.

5. Run rai agent:

   ```bash
   ros2 launch ./src/examples/turtlebot4/turtlebot.launch.py \
       game_launcher:=./src/examples/turtlebot4/simulation/TurtleBot4DemoGamePackage/TurtleBot4Demo.GameLauncher
   ```

   You should be able to see a simulation scene.

   > **TIP**  
   > Press 1, 2 or 3 on the keyboard when simulation window to change the
   > camera. Use W,A,S,D to move the camera.

6. Open you internet browser and go to `localhost:8501`

7. You can interact with your RAI agent through the chat. On the left you can communicate
   with the RAI HMI Agent. On the right you will see status of missions that were send
   to the RAI Node and are executed on your robot.

   - `HMI Agent` is responsible for human-robot interaction.
   - `RAI Node` is responsible for executing tasks on the robot.

   For example you can try the following prompts:

   - Testing robot's identity & RAG:
     - who are you?
     - what is the voltage of the battery and the diameter of the wheels?
   - Testing ROS 2 connection:
     - What are your ros2 interfaces?
     - What do you see?
     - Spin yourself left by 45 degrees and drive 1 meter forward

### Troubleshooting

_My robot doesn't have an identity._

1. Query `rai_whoami_node` for the identity manually

   ```bash
   ros2 service call /rai_whoami_identity_service std_srvs/srv/Trigger
   ```

2. Make sure all required files were generated correctly in the `turtlebot4_whoami` package and have similar sizes to those listed below.

   ```bash
   ls -sh src/examples/turtlebot4_whoami/description/
   total 52K
   4.0K documentation  4.0K images   28K index.faiss  8.0K index.pkl  4.0K robot_constitution.txt  4.0K robot_identity.txt
   ```

   You can also check the contents of `robot_identity.txt` file (it is generated by LLM, but should be similar to the one below).

   ```bash
   cat src/examples/turtlebot4_whoami/description/robot_identity.txt
   I am a TurtleBot® 4, a robotics learning platform designed for education and
   research in robotics. This next-generation mobile robot is built on the iRobot®
   Create® 3 mobile base and comes fully assembled with ROS 2 installed and configured,
   making it accessible for beginners and experienced developers alike.
   ...
   ```

   If files above are incorrect please check your vendor configuration is as described
   in [docs/vendors.md](../../../docs/vendors.md) and rerun point 4 of this tutorial.

---

_Robot doesn't move in the simulation._

1. Make sure you can see ros2 topic simulation binary is running

   1. Run the binary manually:

      ```bash
      cd src/examples/turtlebot4/simulation/
      ./TurtleBot4DemoGamePackage/TurtleBot4Demo.GameLauncher -bg_ConnectToAssetProcessor=0
      ```

   2. Check ros2 topics:

      ```bash
      ros2 topic list
      ```

   3. Verify if Turtlebot simulation binary publishes such topics:

      ```
      /camera_color_info
      /camera_image_color
      /clock
      /cmd_vel
      /parameter_events
      /rosout
      /scan
      /tf
      ```

2. Make sure navigation stack actions work correctly:

   1. Run binary as in the step above.
   2. Run navigation stack:

      ```bash
      cd src/examples/turtlebot4
      ./run-nav.bash
      ```

   3. Check available actions:

      ```bash
      ros2 action list
      ```

   4. Verify if such actions are available:

      ```
      /backup
      /navigate_to_pose
      /spin
      ```

   5. Try to run one of them from command line:

      ```bash
      ros2 action send_goal /spin nav2_msgs/action/Spin "{target_yaw: 3.14}"
      ```
