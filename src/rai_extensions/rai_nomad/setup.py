from setuptools import find_packages, setup

package_name = 'rai_nomad'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['resource/nomad_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kdabrowski',
    maintainer_email='kacper.dabrowski@robotec.ai',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nomad = rai_nomad.nomad:main'
        ],
    },
)
