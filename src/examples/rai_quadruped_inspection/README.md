# Warehouse inspection with Quadruped robot and General Navigation Models exploration

## References

- NoMaD exploration: https://github.com/robodhruv/visualnav-transformer
- Anymal robot: https://www.anybotics.com/robotics/anymal/

## Step-by-step instructions

1. Setup [rai_whoami](../../../docs/create_robots_whoami.md) for `Anymal`:

   ```shell
   source ./setup_shell.sh
   poetry run create_rai_ws --name quadruped --destination-directory src/examples
   ```

2. Add it's documentation, urdf and build robot whoami:

   ```shell
   cd src/examples/quadruped_whoami/description
   cd documentation
   wget https://www.anybotics.com/anymal-technical-specifications.pdf
   cd -
   ```

   ```shell
   mkdir -p urdf
   cd urdf
   wget https://raw.githubusercontent.com/ANYbotics/anymal_c_simple_description/refs/heads/master/urdf/anymal.urdf
   cd -
   ```

   ```shell
   cd rai
   poetry run parse_whoami_package src/examples/quadruped_whoami/description
   colcon build --symlink-install
   ```

3. Place you simulation binary in `sim` folder (refer to [./launch/quadruped.launch.xml](./launch/quadruped.launch.xml)) for exact paths).

4. Run this example

```shell
cd rai
. ./setup_shell.sh
ros2 launch src/examples/rai_quadruped_inspection/launch/quadruped.launch.xml
```

Run `NoMaD`:

> **NOTE** NoMaD is a visual navigation model therefore the generated movement might
> not be optimal/might not be uniformly distributed across whole warehouse area.

```bash
cd rai
. ./setup_shell.sh
ros2 run rai_nomad nomad --ros-args -p image_topic:=/base/camera_image_color
```

Start `NoMaD`:

```shell
ros2 service call /rai_nomad/start std_srvs/srv/Empty
```

Anomalies will be published to `/anomalies` topic:

```shell
ros2 topic echo /anomalies -f
```
