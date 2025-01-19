# Setup RAI with docker

> [!IMPORTANT]
> Docker images are experimental. For tested setup, see the [local setup](../README.md#setup-local).

## 1. Build the docker image

Choose the docker image based on your preferred ROS 2 version.

### 1.1. Humble

```bash
docker build -t rai:humble -f docker/Dockerfile.humble .
```

### 1.2. Jazzy

```bash
docker build -t rai:jazzy -f docker/Dockerfile.jazzy .
```

## 2. Run the docker container

### 2.1. Humble

```bash
docker run -it rai:humble -v $(pwd):/rai
```

### 2.2. Jazzy

```bash
docker run -it rai:jazzy -v $(pwd):/rai
```

## 3. Run the tests to confirm the setup

```sh
cd /rai
source setup_shell.sh
poetry run pytest
```
