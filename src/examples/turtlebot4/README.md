# Turtlebot4 tutorial

The following tutorial will help you get started with [rai](https://github.com/RobotecAI/rai)
using Turtlebot4 simulation.

## Recording

TBD

## Step by step instructions

1. O3DE simulation setup

   1. Download turtlebot4 simulation binary matching your platform from :

      - Ubuntu 22.04 & ros2 humble: [link](https://robotecai-my.sharepoint.com/:u:/g/personal/bartlomiej_boczek_robotec_ai/EZLmGtPNgl9Kiu4royJJnVgB5tjS2Vze0myXDyVJtNcnRw?e=L42Z4z)
      - Ubuntu 24.04 & ros2 jazzy: [link](https://robotecai-my.sharepoint.com/:u:/g/personal/bartlomiej_boczek_robotec_ai/ETkT-jvozlpBtuG1piDeqggBRmWl5eylIChc_g0v_EetpA?e=ina8Dt)

      > **NOTE** If you would like to make changes to the simulation and create your
      > own binary please follow this [README.md](https://github.com/RobotecAI/ROSCon2024Tutorial/README.md)

   2. Unzip it:

      ```bash
      cd rai
      # for Ubuntu 22.04 jammy and ros2 humlbe
      unzip -d src/examples/turtlebot4/simulation Turtlebot4_jammyhumble_0.0.1.zip

      # for Ubuntu 24.04 noble and ros2 jazzy
      unzip -d src/examples/turtlebot4/simulation Turtlebot4_noblejazzy_0.0.1.zip
      ```

2. Clone and install [rai](https://github.com/RobotecAI/rai)

   1. Follow steps from [1. Setting up the workspace](https://github.com/RobotecAI/rai?tab=readme-ov-file#1-setting-up-the-workspace)
   2. Then follow steps from [2. Build the project](https://github.com/RobotecAI/rai?tab=readme-ov-file#2-build-the-project)

3. Setup your LLM vendor: [docs/vendors.md](../../../docs/vendors.md). OpenAI or AWS Bedrock are recommended for now.

4. Configure `rai_whoami_node` (based on ["Your robot identity in RAI"](https://github.com/RobotecAI/rai/blob/development/docs/create_robots_whoami.md) tutorial):

   1. Create `whoami` package for turtlebot4 in `src/examples/turtlebot4`

      ```bash
      . ./setup_shell.sh
      poetry run create_rai_ws --name turtlebot4 --destination-directory src/examples
      ```

   2. Download official turtlebot4 [data sheet](https://bit.ly/3KCp3Du) into
      `src/examples/turtlebot4_whoami/description/documentation`
   3. Download [image](https://s3.amazonaws.com/assets.clearpathrobotics.com/wp-content/uploads/2022/03/16113604/Turtlebot-4-20220207.44.png) of turtlebot4 into `src/examples/turtlebot4_whoami/description/images`
   4. Create robot's identity. Run the `parse_whoami_package`. This will process the documentation, building
      it into a vector database, which is used by RAI agent to reason about its identity.

      > **NOTE**: Vector database is created using the OpenAI API. Parsing bigger documents
      > might lead to costs. Embedding model can be configured in
      > [config.toml](https://github.com/RobotecAI/rai/blob/development/config.toml#L13)

      ```bash
      poetry run parse_whoami_package src/examples/turtlebot4_whoami/description
      # you will be asked to press `y` to continue
      ```

   5. Build the workspace which now includes the new package
      ```bash
      colcon build --symlink-install
      ```

   Congratulations! Your `rai_whoami_node` is configured. In the following steps
   your RAI agent will assume a turtlebot4 "personality".

5. Run rai agent:

   ```bash
   ros2 launch ./src/examples/turtlebot4/turtlebot.launch.xml \
       game_launcher:=./src/examples/turtlebot4/simulation/TurtleBot4DemoGamePackage/TurtleBot4Demo.GameLauncher
   ```

6. Open you internet browser and go to `localhost:8501`

7. You can interact with your RAI agent through the chat. On the left you can communicate
   with the RAI HMI Agent. On the right you will see status of missions that were send
   to the RAI Node and are executed on your robot.

   - `HMI Agent` is responsible for human-robot interaction.
   - `RAI Node` is responsible for executing tasks on the robot.

   For example you can try such prompts:

   - Are you able to bring you something from the kitchen? (testing robot's identity)
   - What are your ros2 interfaces? (discovery of ros2 interfaces)
   - tell me what you see (interpretation of camera image)
   - Drive towards the chair (when table is not visible, robot rejects task that it cannot do)
   - Spin yourself left by 45 degrees (interaction with the robot using ros2 interfaces) - table with the robotic are shoud be visible in the camera
   - Use robotic arm to pick up a box from the table (identity and intefaces doesn't allow it)
   - Drive towards the table (when table is visible, testing ability to interpret camera image and perform actions based on the knowledge)

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

   You can also check the contents of `robot_indentify.txt` file (it is generated by LLM, but should be simillar to the one below).

   ```bash
   cat src/examples/turtlebot4_whoami/description/robot_identity.txt
   I am a TurtleBot® 4, a robotics learning platform designed for education and
   research in robotics. This next-generation mobile robot is built on the iRobot®
   Create® 3 mobile base and comes fully assembled with ROS 2 installed and configured,
   making it accessible for beginners and experienced developers alike.
   ...
   ```

   If files above are incorrect please check your `OPENAI_API_KEY` and rerun point 4
   of this tutorial.

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