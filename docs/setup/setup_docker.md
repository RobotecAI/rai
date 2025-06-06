# Setup RAI with docker

!!! warning "Docker images are experimental"

    Docker images are experimental. For tested setup, see the
    [local setup](install.md).

## 1. Build the docker image

Choose the docker image based on your preferred ROS 2 version.

### 1.1. Humble

```bash
docker build -t rai:humble --build-arg ROS_DISTRO=humble -f docker/Dockerfile .
```

### 1.2. Jazzy

```bash
docker build -t rai:jazzy --build-arg ROS_DISTRO=jazzy -f docker/Dockerfile .
```

## 2. Run the docker container

!!! tip "ROS 2 communication"

    If you intend to run demos on the host machine, ensure the docker container can communicate
    with it. Test this by running the standard ROS 2 example with one node in docker and one on the
    host:
    [link](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html#try-some-examples). If
    topics are not visible or cannot be subscribed to, try using
    [rmw_cyclone_dds](https://github.com/ros2/rmw_cyclonedds) instead of the default
    rmw_fastrtps_cpp.

### 2.1. Humble

```bash
docker run --net=host --ipc=host --pid=host -it rai:humble
```

### 2.2. Jazzy

```bash
docker run --net=host --ipc=host --pid=host -it rai:jazzy
```

## 3. Run the tests to confirm the setup

```sh
cd /rai
source setup_shell.sh
poetry run pytest
```
