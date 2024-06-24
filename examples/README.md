# Running examples

## For ROS 2 example use

This example can be used with real robot or the O3DE simulation.
To run in the simulation it's easiest to use pre-built docker image: see [rai-husarion-demo-private/docker/README.md][husarion_o3de_docker_readme].
Full docs can be found in: [rai-husarion-demo-private/README.md][husarion_o3de_reamde].

```bash
python examples/husarion_poc_example.py
```

## For demo without ROS 2 use

```bash
python examples/agri_example.py
```

In this demo all images are hardcoded.

### Help

```bash
usage: examples/agri_example.py [-h] [--vendor {ollama,openai,awsbedrock}]

Choose the vendor for the scenario runner.

options:
  -h, --help            show this help message and exit
  --vendor {ollama,openai,awsbedrock}
                        Vendor to use for the scenario runner (default: awsbedrock)
```

### Roadmap

- Integration with 2405_vlm demo (ros 2 agri demo): [link](https://github.com/RobotecAI/MultiDomainAgricultureProject/tree/demo/2405_vlm)

[husarion_o3de_reamde]: https://github.com/RobotecAI/rai-husarion-demo-private
[husarion_o3de_docker_readme]: https://github.com/RobotecAI/rai-husarion-demo-private/blob/bb/docker/docker/README.md
