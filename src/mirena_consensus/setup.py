from setuptools import find_packages, setup

package_name = 'mirena_consensus'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='colme',
    maintainer_email='a.colmenero.fernandez@gmail.com',
    description='Sensor Fusion',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mirena_consensus = mirena_consensus.mirena_consensus:main',
        ],
    },
)
