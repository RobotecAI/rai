from setuptools import find_packages, setup

package_name = "rai_grounding_dino"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kajetan Rachwał",
    maintainer_email="kajetan.rachwal@robotec.ai",
    description="Package enabling grounding dino open set detection for RAI",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "grounding_dino = rai_grounding_dino.grounding_dino:main",
            "talker = rai_grounding_dino.talker:main",
        ],
    },
)
