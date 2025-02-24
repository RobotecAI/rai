# How to setup an O3DE scene to work with rai_sim

This page is a tutorial on how to set up your O3DE scene to be compatible with the rai_sim and rai_bench frameworks, i.e. to allow for:

- **spawning** and **despawning** objects
- **tracking their position** in real time

O3DE comes with a ROS2 gem that allows for easy integration of your scene with ROS2. We will use the ROS2 Spawner component and ROS2 Frame component to achieve the desired functionality. The ROS2 Spawner component exposes a service that allows you to spawn and despawn objects in the scene, while the ROS2 Frame component put on an entity will publish its position on the /tf topic.

## Requirements

- https://github.com/o3de/o3de and https://github.com/o3de/o3de-extras checked out on tag **2409.2**
- an o3de project created (instructions: https://www.docs.o3de.org/docs/welcome-guide/create/creating-projects-using-cli/creating-linux/)

## **Step 1. Enable ROS2 gem in your project.**

```bash
scripts/o3de.sh enable-gem -gn ROS2 -pp /path/to/your-project/
```

## **Step 2. Build the Editor and open it**

```bash
cd /path/to/your-project/
cmake -B build/linux -S . -G "Ninja Multi-Config" -DLY_STRIP_DEBUG_SYMBOLS=TRUE -DLY_DISABLE_TEST_MODULES=ON
cmake --build build/linux --config profile --target Editor -j 22
./build/linux/bin/profile/Editor
```

## **Step 3. Set up your spawnable prefabs**

- Place your desired object somewhere in the scene
- On the entity which you want to track, add the ROS2 Frame component and configure it as follows:

![image.png](../imgs/rai_sim/o3de/image.png)

Leave the frame name and joint name empty, set namespace strategy to “Generate from entity name” and check “Publish transform”.

- Save the object as a prefab

![image.png](../imgs/rai_sim/o3de/image%201.png)

![image.png](../imgs/rai_sim/o3de/image%202.png)

You can now delete your entity from the scene.

## **Step 4. Setup ROS2 Spawner**

- Create an empty entity and add a ROS2 spawner component to it.

![image.png](../imgs/rai_sim/o3de/image%203.png)

![image.png](../imgs/rai_sim/o3de/image%204.png)

- Click the “+” icon inside the ROS2 Spawner component window

![image.png](../imgs/rai_sim/o3de/image%205.png)

- Enter the **name by which you will later refer to the object to spawn it**

![image.png](../imgs/rai_sim/o3de/image%206.png)

- Click on the little folder icon that appeared next to your object’s name

![image.png](../imgs/rai_sim/o3de/image%207.png)

- Navigate to the place where you saved your prefab and choose it here and click “OK”

![image.png](../imgs/rai_sim/o3de/image%208.png)

### **Repeat steps 3. and 4. for each object that you want to make spawnable.**

Now, when you press play in the Editor, you should be able to spawn new objects and see their positions on the /tf topic, like so:

```bash
ros2 service call /spawn_entity gazebo_msgs/srv/SpawnEntity "{name: 'apple', initial_pose: {position:{ x: 0.0, y: 0.0, z: 0.2 }, orientation: { x: 0.0, y: 0.0, z: 0.0, w: 1.0 } } }"
```

```bash
ros2 topic echo /tf
```

```bash
kdabrowski@robo-pc-019:~$ ros2 topic echo /tf
transforms:
- header:
    stamp:
      sec: 163
      nanosec: 66984981
    frame_id: apple_1/odom
  child_frame_id: apple_1/
  transform:
    translation:
      x: -2.6995063308277167e-05
      y: -2.1816253138240427e-05
      z: 0.0499994121491909
    rotation:
      x: 2.1038686099927872e-05
      y: -5.1415558118605986e-05
      z: -3.404247763683088e-05
      w: 1.0
---
```

If these commands work for you, your scene is successfully set up and ready to be used with rai_sim. You can now build the binary package and run it with rai_sim.

## Step 5. Building the binary package

```bash
cmake -B build/linux_mono -S . -G "Ninja Multi-Config" -DLY_MONOLITHIC_GAME=1
cmake --build build/linux_mono --config release --target install -j 22
```

The package should now be ready to run in the `install/bin/Linux/release/Monolithic` directory inside your project.
