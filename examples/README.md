# Running examples

## For ROS 2 example use

```bash
python examples/husarion_poc_example.py
```

## For demo without ROS 2 use

```bash
python examples/agri_example.py
```

In this demo all images are hardcoded.

## For husarion The Describer demo

Husarion is supposed to roam the environment and describe what it sees. \
The observations are saved to map_database.txt in the format of `x: {}, y: {}, z: {}, observation"`
The demo only works with bedrock for now.

```bash
python examples/explore_and_describe_bedrock.py
```

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
