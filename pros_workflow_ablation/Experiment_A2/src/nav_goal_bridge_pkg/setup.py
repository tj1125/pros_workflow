from glob import glob

from setuptools import find_packages, setup

package_name = "nav_goal_bridge_pkg"
launch_files = glob("launch/*.launch.py") + glob("launch/*.xml")
config_files = glob("config/*.yaml") + glob("config/*.pgm")

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", launch_files),
        ("share/" + package_name + "/config", config_files),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="VLM Team",
    maintainer_email="vlm@project.local",
    description="Bridge /goal_pose to Nav2 and pros-style navigation control",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "nav_goal_bridge_node = nav_goal_bridge_pkg.main:main",
            "scan_throttle_node = nav_goal_bridge_pkg.scan_throttle:main",
        ],
    },
)
