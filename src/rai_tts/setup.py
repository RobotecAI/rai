import os
from glob import glob

from setuptools import find_packages, setup

package_name = "rai_tts"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="maciejmajek",
    maintainer_email="maciej.majek@robotec.ai",
    description="A Text To Speech package with streaming capabilities.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "tts_node = rai_tts.tts_node:main",
        ],
    },
)
