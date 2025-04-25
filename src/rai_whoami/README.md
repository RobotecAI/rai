# RAI Whoami

## Description

The RAI Whoami is a package providing tools to extract information about a robot from a given directory.
Including pdf, docx, doc, md, urdf files and images.

## Creating a whoami configuration for a robot

### Prerequisites

- A directory with the following structure:

```
documentation_dir/
├── images/ # png, jpg, jpeg files
├── documentation/ # pdf, docx, doc, md files
├── urdf/ # urdf files
```

### Building the whoami configuration

```bash
python src/rai_whoami/rai_whoami/build_whoami.py documentation_dir --output_dir output_dir
```

### Using the whoami configuration

```python
from rai_whoami import EmbodimentInfo

info = EmbodimentInfo.from_directory("path/to/output_dir")
```
