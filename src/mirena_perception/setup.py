from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'mirena_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Register the package in the ROS2 index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Include the package.xml file
        ('share/' + package_name, ['package.xml']),
        # Include launch files if you have them in a /launch folder
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
        'matplotlib'
    ],
    zip_safe=True,
    maintainer='Ozuba',
    maintainer_email='ozuba@ozuba.es',
    description='Perception node',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # This allows you to run: ros2 run mirena_planner planner_node
            'mirena_perception= mirena_perception.mirena_perception:main',
            'perception_calibration = mirena_perception.perception_calibration:main',
        ],
    },
)