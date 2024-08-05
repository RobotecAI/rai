from setuptools import find_packages, setup

package_name = "rai_hmi"

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
    maintainer="Adam Gotlib",
    maintainer_email="adam.gotlib@robotec.ai",
    description="A Human-Machine Interface (HMI) to converse with the robot.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "hmi_node = rai_hmi.hmi_node:main",
        ],
    },
)
