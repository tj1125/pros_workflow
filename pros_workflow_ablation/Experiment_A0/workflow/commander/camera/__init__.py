from .capture import get_camera_image_base64, get_camera_rgbd_base64


def configured_room_cameras():
    from .registry import configured_room_cameras as _configured_room_cameras

    return _configured_room_cameras()


def room_camera_topic(camera_name: str):
    from .registry import room_camera_topic as _room_camera_topic

    return _room_camera_topic(camera_name)


__all__ = [
    "configured_room_cameras",
    "get_camera_image_base64",
    "get_camera_rgbd_base64",
    "room_camera_topic",
]
