# Your robot identity in RAI

RAI Agent needs to understand what kind of robot it is running on.
This includes its looks, purpose, ethical code, equipment, capabilities and documentation.
To configure RAI for your robot, provide contents for your robot's so called `whoami` package.

Your robot's `whoami` package serves as a configuration package for the `rai_whoami` node.

## Example (Franka Emika Panda arm)

1. Setup the repository using 1st and 2nd step from [Setup](../README.md#setup)

2. Create a whoami package for Panda

   ```shell
   poetry run create_rai_ws --name panda --destination-directory src/examples
   ```

3. Fill in the `src/examples/panda_whoami/description` folder with data:\
   2.1 Save [this image](https://robodk.com/robot/img/Franka-Emika-Panda-robot.png) into `src/examples/panda_whoami/description/images`\
   2.2 Save [this document](https://github.com/user-attachments/files/16417196/Franka.Emika.Panda.robot.-.RoboDK.pdf) in `src/examples/panda_whoami/description/documentation`

4. Run the `parse_whoami_package`. This will process the documentation, building it into a vector database, which is used by RAI agent to reason about its identity.

> **NOTE**: By default the vector database is created using the OpenAI API. Parsing
> bigger documents might lead to costs. Embedding model can be configured in
> [config.toml](../config.toml) (`ollama` works locally, see [docs/vendors.md](./vendors.md#ollama)).

```shell
poetry run parse_whoami_package src/examples/panda_whoami/description
```

> [!IMPORTANT]  
> For now, this works only if you have OPENAI_API_KEY variable exported.

## Testing

You can test your new `panda_whoami` package by calling `rai_whoami` services:

2. Building the `rai_whoami` package and running the `rai_whoami_node` for your `Panda` robot:

```shell
colcon build --symlink-install
ros2 run rai_whoami rai_whoami_node --ros-args -p robot_description_package:="panda_whoami"
```

2. Calling the rai_whoami services

```shell
ros2 service call /rai_whoami_identity_service std_srvs/srv/Trigger # ask for identity
ros2 service call /rai_whoami_selfimages_service std_srvs/srv/Trigger # ask for images folder
ros2 service call /rai_whoami_constitution_service std_srvs/srv/Trigger # ask for robot constitution
ros2 service call /rai_whoami_documentation_service rai_interfaces/srv/VectorStoreRetrieval "query:  'maximum load'" # ask for Panda's maximum load
```

If your service calls succeed, your `panda_whoami` package has been properly initialized.
