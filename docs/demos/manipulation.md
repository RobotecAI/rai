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
    poetry install --with perception,simbench
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

1. Run the Demo:

    ```shell
    streamlit run examples/manipulation-demo-streamlit.py
    ```

    Alternatively, you can run the simpler command-line version, which also serves as an example of
    how to use the RAI API for your own applications:

    1. Run Simulation

    ```shell
    ros2 launch examples/manipulation-demo.launch.py game_launcher:=demo_assets/manipulation/RAIManipulationDemo/RAIManipulationDemo.GameLauncher
    ```

    2. Run cmd app

    ```shell
    python examples/manipulation-demo.py
    ```

2. Interact with the robot arm using natural language commands. For example:

    ```
    "Place every apple on top of the cube"
    "Build a tower from cubes"
    "Arrange objects in a line"
    "Put two boxes closer to each other. Move only one box."
    "Move cubes to the left side of the table"
    ```

!!! tip "Changing camera view"

    To change camera in the simulation use 1-7 keys on your keyboard once it's window is focused.

### Docker Setup

#### 1. Setting up the demo

1.  Set up docker as outlined in the [docker setup guide](../setup/setup_docker.md). During the setup, build the docker image with all dependencies (i.e., use the `--build-arg DEPENDENCIES=all_groups` argument)

2.  Enable X11 access for the docker container:

    ```shell
    xhost +local:root
    ```

3.  Run the docker container with the following command:

    ```shell
    docker run --net=host --ipc=host --pid=host -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --gpus all -it rai:jazzy # or rai:humble
    ```

    !!! tip "NVIDIA Container Toolkit"

        In order to use the `--gpus all` flag, the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) must be installed on the host machine.

4.  (Inside the docker container) By default, RAI uses OpenAI as the vendor. Thus, it is necessary
    to set the `$OPENAI_API_KEY` environmental variable. The command below may be utilized to set
    the variable and add it to the container's `.bashrc` file:

    ```shell
    export OPENAI_API_KEY=YOUR_OPEN_AI_API_KEY
    echo "export OPENAI_API_KEY=$OPENAI_API_KEY" >> ~/.bashrc
    ```

    !!! note AI vendor change

        The default vendor can be changed to a different provider via the [RAI configuration tool](../setup/install.md#15-configure-rai)

5.  After this, follow the steps in the [Local Setup](#local-setup) from step 2 onwards.

    !!! tip "New terminal in docker"

        In order to open a new terminal in the same docker container, you can use the following command:

        ```shell
        docker exec -it <container_id> bash
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
