from setuptools import find_packages, setup

package_name = "bfmc_v2x"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="Stefan Oanea",
    maintainer_email="stefan.oanea@gmail.com",
    description="V2X localisation and cooperative driving logic, free of ROS.",
    tests_require=["pytest"],
)
