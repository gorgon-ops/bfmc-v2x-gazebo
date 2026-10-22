import os
from glob import glob

from setuptools import find_packages, setup

package_name = "bfmc_v2x_ros"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Stefan Oanea",
    maintainer_email="stefan.oanea@gmail.com",
    description="ROS 2 nodes for the BFMC V2X localisation demonstration.",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "localization_server = bfmc_v2x_ros.localization_server_node:main",
            "semaphores = bfmc_v2x_ros.semaphore_node:main",
            "wheel_encoder = bfmc_v2x_ros.wheel_encoder_node:main",
            "v2x_bridge = bfmc_v2x_ros.v2x_bridge_node:main",
            "localization = bfmc_v2x_ros.localization_node:main",
            "driver = bfmc_v2x_ros.driver_node:main",
            "detector = bfmc_v2x_ros.detector_node:main",
            "environment_server = bfmc_v2x_ros.environment_server_node:main",
            "metrics = bfmc_v2x_ros.metrics_node:main",
        ],
    },
)
