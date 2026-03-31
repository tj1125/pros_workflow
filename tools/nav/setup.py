from glob import glob

from setuptools import setup

package_name = "vlm_rl_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (
            f"share/{package_name}",
            [
                "package.xml",
            ],
        ),
        (f"share/{package_name}/launch", ["navigation.launch.py"]),
        (f"share/{package_name}/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="VLM-RL Team",
    maintainer_email="vlm_rl@project.local",
    description="VLM-RL self-contained localization and planner package",
    license="MIT",
    entry_points={
        "console_scripts": [],
    },
)
