from glob import glob

from setuptools import find_packages, setup

package_name = "nav_goal_bridge_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*")),
        ("share/" + package_name + "/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="VLM-RL Team",
    maintainer_email="vlm_rl@project.local",
    description="Bridge /goal_pose to Nav2 and pros-style navigation control",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "nav_goal_bridge_node = nav_goal_bridge_pkg.main:main",
        ],
    },
)
