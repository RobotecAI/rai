# Your robot identity in RAI

RAI Agent needs to understand what kind of robot it is running on.
This includes its looks, purpose, ethical code, equipment, capabilities and documentation.
To configure RAI for your robot, provide contents for your robot's so called `whoami` package.

Your robot's `whoami` package serves as a configuration package for the `rai_whoami` node.

> [!TIP]
> The Human-Machine Interface (HMI), both text and voice versions, relies heavily on the whoami package. It uses the robot's identity, constitution, and documentation to provide context-aware responses and ensure the robot behaves according to its defined characteristics.

## Configuration example - Franka Emika Panda arm

1. Setup the repository using 1st and 2nd step from [Setup](../README.md#setup)

2. Create a whoami package for Panda

   ```shell
   ./scripts/create_rai_ws.sh --name panda --destination-directory src/examples
   ```

3. Fill in the `src/examples/panda_whoami/description` folder with data:

   3.1. Save [this image](https://robodk.com/robot/img/Franka-Emika-Panda-robot.png) into `src/examples/panda_whoami/description/images`

   3.2. Save [this document](https://github.com/user-attachments/files/16417196/Franka.Emika.Panda.robot.-.RoboDK.pdf) in `src/examples/panda_whoami/description/documentation`

   3.3. Save [this urdf](https://github.com/frankaemika/franka_ros/blob/develop/franka_description/robots/panda/panda.urdf.xacro) in `src/examples/panda_whoami/description/urdf`

4. Run the `parse_whoami_package`. This will process the documentation, building it into a vector database, which is used by RAI agent to reason about its identity.

> [!IMPORTANT]
> Parsing bigger documents with Cloud vendors might lead to costs. Consider using the
> local `ollama` provider for this task. Embedding model can be configured in
> [config.toml](../config.toml) (`ollama` works locally, see [docs/vendors.md](./vendors.md#ollama)).

```shell
./scripts/parse_whoami_package.sh src/examples/panda_whoami
```

5. Optional: Examine the generated files

After running the `parse_whoami_package` command, you can inspect the generated files in the `src/examples/panda_whoami/description/generated` directory. These files contain important information about your robot:

- `robot_identity.txt`: Contains a detailed description of the robot's identity, capabilities, and characteristics.
- `robot_description.urdf.txt`: Provides a summary of the robot's URDF (Unified Robot Description Format), describing its physical structure.
- `robot_constitution.txt`: Outlines the ethical guidelines and operational rules for the robot.
- `faiss_index`: A directory containing the vector store of the robot's documentation, used for efficient information retrieval.

## Testing

You can test your new `panda_whoami` package by calling `rai_whoami` services:

1. Building the `rai_whoami` package and running the `rai_whoami_node` for your `Panda` robot:

```shell
colcon build --symlink-install
ros2 run rai_whoami rai_whoami_node --ros-args -p robot_description_package:="panda_whoami"
```

2. Calling the rai_whoami services

```shell
ros2 service call /rai_whoami_identity_service std_srvs/srv/Trigger # ask for identity
ros2 service call /rai_whoami_selfimages_service std_srvs/srv/Trigger # ask for images folder
ros2 service call /rai_whoami_constitution_service std_srvs/srv/Trigger # ask for robot constitution
ros2 service call /rai_whoami_urdf_service std_srvs/srv/Trigger # ask for urdf description
ros2 service call /rai_whoami_documentation_service rai_interfaces/srv/VectorStoreRetrieval "query:  'maximum load'" # ask for Panda's maximum load
```

If your service calls succeed, your `panda_whoami` package has been properly initialized.
