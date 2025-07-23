# Manipulation tasks with natural language

This demo showcases the capabilities of RAI in performing manipulation tasks using natural language
commands. The demo utilizes a robot arm (Franka Emika Panda) in a simulated environment,
demonstrating how RAI can interpret complex instructions and execute them using advanced vision and
manipulation techniques.

![Manipulation Demo](../imgs/manipulation_demo.gif)

## Setup

> [!TIP] LLM model
>
> The demo uses the `complex_model` LLM configured in `config.toml`. The model should be a
> multimodal, tool-calling model. See [Vendors](../setup/vendors.md#llm-model-configuration-in-rai).

!!! tip "ROS 2 Sourced"

    Make sure ROS 2 is sourced. (e.g. `source /opt/ros/humble/setup.bash`)

### Local Setup

#### Setting up the demo

1. Follow the RAI setup instructions in the [quick setup guide](../setup/install.md#setting-up-developer-environment).
2. Download additional dependencies:

    ```shell
    poetry install --with openset
    vcs import < demos.repos
    rosdep install --from-paths src/examples/rai-manipulation-demo/ros2_ws/src --ignore-src -r -y
    ```

3. Download the latest binary release

    ```bash
    ./scripts/download_demo.sh manipulation
    ```

4. Build the ROS 2 workspace:

    ```bash
    colcon build --symlink-install
    ```

#### Running the demo

!!! note "Remain in sourced shell"

    Ensure that every command is run in a sourced shell using `source setup_shell.sh`
    Ensure ROS 2 is sourced.

1. Start the demo

    ```shell
    ros2 launch examples/manipulation-demo.launch.py game_launcher:=demo_assets/manipulation/RAIManipulationDemo/RAIManipulationDemo.GameLauncher
    ```

2. In the second terminal, run the streamlit interface:

    ```shell
    streamlit run examples/manipulation-demo-streamlit.py
    ```

    Alternatively, you can run the simpler command-line version, which also serves as an example of
    how to use the RAI API for you own applications:

    ```shell
    python examples/manipulation-demo.py
    ```

3. Interact with the robot arm using natural language commands. For example:

    ```
    Enter a prompt: Pick up the red cube and drop it on another cube
    ```

!!! tip "Changing camera view"

    To change camera in the simulation use 1-7 keys on your keyboard once it's window is focused.

### Docker Setup

!!! note "ROS 2 required"

    The docker setup requires a working Humble or Jazzy ROS 2 installation on the host machine. Make sure that ROS 2 is sourced on the host machine and the `ROS_DOMAIN_ID` environment variable is set to the same value as in the [Docker setup](../setup/setup_docker.md#2-set-up-communications-between-docker-and-host-optional)

!!! warning "ROS 2 distributions"

    It is highly recommended that ROS 2 distribution on the host machine matches the ROS 2 distribution of the docker container. A distribution version mismatch may result in the demo not working correctly.

#### 1. Setting up the demo

1.  Set up docker as outlined in the [docker setup guide](../setup/setup_docker.md). During the setup, build the docker image with all dependencies (i.e., use the `--build-arg DEPENDENCIES=all_groups` argument)
    and configure communication between the container and the host ([link](../setup/setup_docker.md#2-set-up-communications-between-docker-and-host-optional)).

2.  On the host machine, download the latest binary release for the Robotic Arm Demo:

    ```shell
    ./scripts/download_demo.sh manipulation
    ```

3.  Run the docker container (if not already running):

    ```shell
    docker run --net=host --ipc=host --pid=host -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID -it rai:jazzy # or rai:humble
    ```

    !!! tip "NVIDIA GPU acceleration"

        If the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is set up on your host machine, you can use the GPU within the RAI docker container for faster inference by adding the `--gpus all` option:

        ```shell
        docker run --net=host --ipc=host --pid=host -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID --gpus all -it rai:jazzy # or rai:humble
        ```

        Sometimes, passing GPUs to the docker container may result in an error:

        ```shell
        docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
        ```

        Restarting the docker service should resolve this error:

        ```shell
        sudo systemctl restart docker
        ```

4.  (Inside the container shell) Download additional ROS 2 dependencies:

    ```shell
    vcs import < demos.repos
    rosdep install --from-paths src/examples/rai-manipulation-demo/ros2_ws/src --ignore-src -r -y
    ```

5.  (Inside the container shell) Build the ROS 2 workspace:

    ```shell
    source /opt/ros/${ROS_DISTRO}/setup.bash
    colcon build --symlink-install
    ```

#### 2. Running the demo

!!! note Source the setup shell

    Ensure ROS 2 is sourced on the host machine and the `ROS_DOMAIN_ID` environment variable is set to the same value as in the [Docker setup](../setup/setup_docker.md#2-set-up-communications-between-docker-and-host-optional). Ensure that every command inside the docker container is run in a sourced shell using `source setup_shell.sh`.

1. Launch the Robotic Arm Visualization on the host machine:

    ```shell
    ./demo_assets/manipulation/RAIManipulationDemo/RAIManipulationDemo.GameLauncher
    ```

2. (Inside the container shell) Launch the Robotic Arm Demo script inside of the docker container:

    ```shell
    ros2 launch examples/manipulation-demo.launch.py
    ```

3. (Inside the container shell) Open a new terminal for the docker container (e.g., `docker exec -it CONTAINER_ID /bin/bash`) and launch the streamlit interface:

    ```shell
    streamlit run examples/manipulation-demo-streamlit.py
    ```

    Alternatively, run the simpler command-line version:

    ```shell
    python examples/manipulation-demo.py
    ```

## How it works

The manipulation demo utilizes several components:

1. Vision processing using Grounded SAM 2 and Grounding DINO for object detection and segmentation.
2. RAI agent to process the request and plan the manipulation sequence.
3. Robot arm control for executing the planned movements.

The main logic of the demo is implemented in the `create_agent` function, which can be found in:

```python
examples/manipulation-demo.py
```

## Known Limitations

-   `Grounding DINO` can't distinguish colors.
-   VLMs tend to struggle with spatial understanding (for example left/right concepts).

!!! tip "Building from source"

    If you are having trouble running the binary, you can build it from source
    [here](https://github.com/RobotecAI/rai-manipulation-demo).
