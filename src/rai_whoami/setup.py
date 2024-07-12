from setuptools import find_packages, setup

package_name = "rai_whoami"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Adam Dabrowski",
    maintainer_email="adam.dabrowski@robotec.ai",
    description="RAI package wrapping up sources of information about the robot, and presenting interfaces to interact with it",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["rai_whoami_node = rai_whoami.rai_whoami_node:main"],
    },
)
