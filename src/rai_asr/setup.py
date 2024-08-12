import os
from glob import glob

from setuptools import find_packages, setup

package_name = "rai_asr"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mkotynia",
    maintainer_email="magdalena.kotynia@robotec.ai",
    description="An Automatic Speech Recognition package, leveraging Whisper for transcription and \
        Silero VAD for voice activity detection. This node captures audio, detects speech, and \
        transcribes the spoken content, publishing the transcription to a topic. ",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["asr_node = rai_asr.asr_node:main"],
    },
)
