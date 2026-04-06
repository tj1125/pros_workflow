from glob import glob
import os

from setuptools import find_packages, setup


package_name = "robot_description"


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*.urdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="Robot description assets for the arm IK controller.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
