from setuptools import setup
from glob import glob
import os

package_name = "vlm_rl_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        # ament resource index
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        # package.xml
        (f"share/{package_name}", ["package.xml"]),
        # launch files
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        # config files
        (f"share/{package_name}/config", glob("config/*.yaml")),
        # map files
        (f"share/{package_name}/map", glob("map/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="VLM-RL Team",
    maintainer_email="vlm_rl@project.local",
    description="VLM-RL self-contained Nav2 navigation package",
    license="MIT",
    entry_points={
        "console_scripts": [],
    },
)
