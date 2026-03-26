import re
from pathlib import Path
from typing import List, Optional

import yaml


_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "cameras.yaml"
_ROOM1_CAMERA_RE = re.compile(r"^Camera_Room1_(\d+)$")
_GROUP_SIZE = 3


def room1_camera_group_id(camera_name: str) -> Optional[int]:
    """Return the 1-based room camera group ID for Camera_Room1_<n> names."""
    match = _ROOM1_CAMERA_RE.match(str(camera_name).strip())
    if not match:
        return None

    camera_idx = int(match.group(1))
    if camera_idx < 1:
        return None

    return ((camera_idx - 1) // _GROUP_SIZE) + 1


def _load_camera_names() -> List[str]:
    with open(_CONFIG_PATH, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
        cameras = payload.get("cameras", [])
    return [str(camera.get("name", "")).strip() for camera in cameras if camera.get("name")]


def configured_room_cameras() -> List[str]:
    """Return all configured fixed room cameras in config order."""
    return [
        camera_name
        for camera_name in _load_camera_names()
        if room1_camera_group_id(camera_name) is not None
    ]


def cameras_in_group(group_id: int) -> List[str]:
    """Return configured room cameras that belong to the given 1-based group."""
    if group_id < 1:
        return []

    return [
        camera_name
        for camera_name in _load_camera_names()
        if room1_camera_group_id(camera_name) == group_id
    ]


def group_cameras_for_camera(camera_name: str) -> List[str]:
    """Resolve the configured room-camera group for a specific camera name."""
    group_id = room1_camera_group_id(camera_name)
    if group_id is None:
        return []
    return cameras_in_group(group_id)
