# Setup RAI with docker

!!! warning "Docker images are experimental"

    Docker images are experimental. For tested setup, see the
    [local setup](install.md).

## 1. Get the docker image

Images are built from `docker/Dockerfile` on `ros:<distro>-ros-base` for ROS 2 Jazzy and Lyrical (amd64 only). They contain every dependency group except `docs`, `rai_interfaces`, jackd and PortAudio. The repository is in `/rai`, with the venv in `/rai/.venv`.

The CI image (CPU-only torch) is published to GitHub Container Registry:

```bash
docker pull ghcr.io/robotecai/rai-dev:lyrical  # or jazzy
```

To build it yourself, clone the repository and pick a target: `dev` (CUDA torch, the default) or `ci` (CPU-only torch).

```bash
git clone https://github.com/RobotecAI/rai.git
cd rai
docker build -t rai:lyrical --build-arg ROS_DISTRO=lyrical --target dev -f docker/Dockerfile .
docker build -t rai:jazzy --build-arg ROS_DISTRO=jazzy --target dev -f docker/Dockerfile .
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
    source /opt/ros/lyrical/setup.bash # or jazzy
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

The entrypoint sources ROS 2, the workspace and the venv.

```bash
docker run -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID -it rai:lyrical
```

## 4. Run the tests to confirm the setup

```sh
docker/virtual-audio.sh  # virtual audio devices for the sound tests
pytest
```
