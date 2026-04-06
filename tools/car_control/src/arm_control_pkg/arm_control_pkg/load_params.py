from pathlib import Path

import yaml

try:
    from ament_index_python.packages import (
        PackageNotFoundError,
        get_package_share_directory,
    )
except ImportError:  # pragma: no cover - available in ROS runtime
    PackageNotFoundError = Exception
    get_package_share_directory = None


class LoadParams:
    def __init__(self, package_name: str, config_filename: str = "arm_config.yaml"):
        self.package_name = package_name
        self.config_filename = config_filename
        self.arm_params = {}
        self._load_arm_parameters()

    def _resolve_config_path(self) -> Path:
        candidates = []

        if get_package_share_directory is not None:
            try:
                share_dir = Path(get_package_share_directory(self.package_name))
                candidates.append(share_dir / "config" / self.config_filename)
            except PackageNotFoundError:
                pass

        candidates.append(Path(__file__).resolve().parents[1] / "config" / self.config_filename)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Unable to locate {self.config_filename} for package {self.package_name}."
        )

    def _load_arm_parameters(self) -> None:
        yaml_file_path = self._resolve_config_path()
        with yaml_file_path.open("r", encoding="utf-8") as file:
            self.arm_params = yaml.safe_load(file)

    def get_arm_params(self):
        return self.arm_params
