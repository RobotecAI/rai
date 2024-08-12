# robot_whoami

Certain parts of RAI rely on robot's whoami package.

## Creating robot's whoami (Franka Emika Panda)

1. Create a whoami package for panda

```
poetry run create_rai_ws --name panda --destination-directory src/examples
```

2. Fill in the src/examples/panda_whoami/description folder with data:\
   2.1 Save [this image](https://robodk.com/robot/img/Franka-Emika-Panda-robot.png) into `src/examples/panda_whoami/description/images`\
   2.2 Save [this document](https://github.com/user-attachments/files/16417196/Franka.Emika.Panda.robot.-.RoboDK.pdf) in `src/examples/panda_whoami/description/documentation`

3. Run the parse_whoami_package to build vector database and reason out the identity

```
poetry run parse_whoami_package src/examples/panda_whoami/description src/examples/panda_whoami/description
```

> [!IMPORTANT]  
> For now, this works only if you have OPENAI_API_KEY variable exported.

## Testing panda_whoami

rai_whoami provides services for gathering information about current platform. Test the panda_whoami package by:

1. Building and sourcing the install

```
colcon build
source install/setup.sh
export PYTHONPATH="$(dirname $(dirname $(poetry run which python)))/lib/python$(poetry run python --version | awk '{print $2}' | cut -d. -f1,2)/site-packages:$PYTHONPATH"
```

2. Calling the rai_whoami services

```
ros2 service call /rai_whoami_identity_service std_srvs/srv/Trigger # ask for identity
ros2 service call /rai_whoami_selfimages_service std_srvs/srv/Trigger # ask for images folder
ros2 service call /rai_whoami_constitution_service std_srvs/srv/Trigger # ask for robot constitution
ros2 service call /rai_whoami_documentation_service rai_interfaces/srv/VectorStoreRetrieval "query:  'maximum load'" # ask for panda's maximum load
```
