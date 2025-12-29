# Setup RAI with docker

!!! warning "Docker images are experimental"

    Docker images are experimental. For tested setup, see the
    [local setup](install.md).

## 1. Build the docker image

Choose the docker image based on your preferred ROS 2 version. You may build the selected image with only the core dependencies or, alternatively, with all the additional modules. To build the docker image, you must clone the RAI repository:

```bash
git clone https://github.com/RobotecAI/rai.git
cd rai
```

### 1.1. Humble

Core dependencies only:

```bash
docker build -t rai:humble --build-arg ROS_DISTRO=humble -f docker/Dockerfile .
```

All dependencies:

```bash
docker build -t rai:humble --build-arg ROS_DISTRO=humble --build-arg DEPENDENCIES=all_groups -f docker/Dockerfile .
```

### 1.2. Jazzy

Core dependencies only:

```bash
docker build -t rai:jazzy --build-arg ROS_DISTRO=jazzy -f docker/Dockerfile .
```

All dependencies:

```bash
docker build -t rai:jazzy --build-arg ROS_DISTRO=jazzy --build-arg DEPENDENCIES=all_groups -f docker/Dockerfile .
```

## 2. Set up communications between docker and host (Optional)

!!! tip "ROS 2 communication"

    If you intend to run demos on the host machine, ensure the docker container can communicate
    with it. Test this by running the standard ROS 2 example with one node in docker and one on the
    host:
    [link](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html#try-some-examples).

!!! warning "ROS 2 distributions"

    It is highly recommended that ROS 2 distribution on the host machine matches the ROS 2 distribution of the docker container. A distribution version mismatch may result in the demos not working correctly.

To allow the container to communicate with the host machine, configure the host environment as presented below:

1. Source ROS 2 on the host machine:

    ```shell
    source /opt/ros/jazzy/setup.bash # or humble
    ```

2. If not configured, set the `ROS_DOMAIN_ID` environment variable to a domain ID between 0 and 101, inclusive. Example:

    ```shell
    export ROS_DOMAIN_ID=99
    ```

3. Install the eProsima Fast DDS middleware (should come preinstalled with ROS 2):

    ```shell
    sudo apt install ros-"${ROS_DISTRO}"-fastrtps
    ```

4. Configure the DDS middleware using the `fastrtps_config.xml` file included in the RAI repository:

    ```shell
    export FASTRTPS_DEFAULT_PROFILES_FILE=$(pwd)/docker/fastrtps_config.xml

    ```

5. Set the RMW to use eProsima Fast DDS:

    ```shell
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    ```

## 3. Run the docker container

### 3.1. Humble

```bash
docker run --net=host --ipc=host --pid=host -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID -it rai:humble
```

### 3.2. Jazzy

```bash
docker run --net=host --ipc=host --pid=host -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID -it rai:jazzy
```

## 4. Run the tests to confirm the setup

```sh
cd /rai
source setup_shell.sh
poetry run pytest tests/{agents,messages,tools,types}
```
