from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import scipy.io as sio
from PIL import Image as PilImage
import rclpy
from rclpy.node import Node
from rclpy.subscription import Subscription
from sensor_msgs.msg import CameraInfo, CompressedImage


@dataclass(frozen=True)
class CameraConfig:
    name: str
    rgb_topic: str
    depth_topic: str
    meta_topic: str
    capture_depth: bool


class ImagePairSubscriber(Node):
    """Capture sequential RGB/Depth/Meta triplets for multiple cameras."""

    def __init__(self) -> None:
        super().__init__('image_pair_subscriber')

        default_output_dir = Path('../pics')

        self.declare_parameter('camera_names', ['1_1', '1_2', '1_3'])
        self.declare_parameter('rgb_topic_template', '/rgb_{camera}/image/compressed')
        self.declare_parameter('depth_topic_template', '/depth_{camera}/depth/compressed')
        self.declare_parameter('meta_topic_template', '/camera_info_{camera}/camera_info')
        self.declare_parameter('output_dir', str(default_output_dir))
        self.declare_parameter('timestamp_format', '%Y%m%d_%H%M%S_%f')
        self.declare_parameter('cooldown_seconds', 3.0)
        self.declare_parameter('sensor_size_mm', [3.896, 2.453])
        self.declare_parameter('near_clip', 0.3)
        self.declare_parameter('far_clip', 12.0)

        self.timestamp_format = str(self.get_parameter('timestamp_format').value)
        self.cooldown_seconds = float(self.get_parameter('cooldown_seconds').value)
        self.sensor_size_mm = self._read_sensor_size_param('sensor_size_mm')
        self.near_clip = float(self.get_parameter('near_clip').value)
        self.far_clip = float(self.get_parameter('far_clip').value)

        output_dir_param = str(self.get_parameter('output_dir').value)
        self.base_dir = Path(output_dir_param).expanduser().resolve()
        self.rgb_dir = self.base_dir / 'rgb'
        self.depth_dir = self.base_dir / 'depth'
        self.meta_dir = self.base_dir / 'camera_parameter'
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.meta_recorded: Set[str] = set()

        camera_names = self._read_camera_names()
        rgb_template = str(self.get_parameter('rgb_topic_template').value)
        depth_template = str(self.get_parameter('depth_topic_template').value)
        meta_template = str(self.get_parameter('meta_topic_template').value)

        self.camera_configs: List[CameraConfig] = []
        for name in camera_names:
            rgb_topic = rgb_template.format(camera=name)
            depth_topic = depth_template.format(camera=name)
            meta_topic = meta_template.format(camera=name)
            capture_depth = self._requires_depth_capture(rgb_topic, depth_topic, name)
            self.camera_configs.append(
                CameraConfig(
                    name=name,
                    rgb_topic=rgb_topic,
                    depth_topic=depth_topic,
                    meta_topic=meta_topic,
                    capture_depth=capture_depth,
                )
            )
        if not self.camera_configs:
            raise ValueError('At least one camera name is required.')

        for cfg in self.camera_configs:
            if self._meta_file_path(cfg.name).exists():
                self.meta_recorded.add(cfg.name)

        self.current_camera_index = 0
        self.stage: str = 'capture'
        self.current_timestamp_label: Optional[str] = None
        self.rgb_captured = False
        self.depth_captured = False
        self.current_capture_depth = False
        self.cooldown_timer = None
        self._active_subscriptions: Dict[str, Subscription] = {}
        self._activate_camera_subscriptions(self.active_camera)

        config_strings = ', '.join(
            f'{cfg.name} -> (rgb:{cfg.rgb_topic}, depth:{cfg.depth_topic}, meta:{cfg.meta_topic})'
            for cfg in self.camera_configs
        )
        self.get_logger().info(
            f'Output directory: {self.base_dir}; cooldown={self.cooldown_seconds}s\nCameras: {config_strings}'
        )

    def _read_camera_names(self) -> List[str]:
        names_param = self.get_parameter('camera_names').value
        if isinstance(names_param, list):
            names = [str(name) for name in names_param if str(name).strip()]
        else:
            names = [str(names_param).strip()]
        return [name for name in names if name]

    def _read_sensor_size_param(self, name: str) -> np.ndarray:
        raw = self.get_parameter(name).value
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(f'{name} must be a list with two entries (width_mm, height_mm).')
        data = np.array(raw, dtype=np.float64)
        if np.any(data <= 0):
            raise ValueError(f'{name} entries must be positive numbers.')
        return data

    @property
    def active_camera(self) -> CameraConfig:
        return self.camera_configs[self.current_camera_index]

    def _on_rgb(self, camera_name: str, msg: CompressedImage) -> None:
        if not self._should_handle(camera_name, 'rgb'):
            return
        if self.rgb_captured:
            return

        if self.current_timestamp_label is None:
            self.current_timestamp_label = datetime.now().strftime(self.timestamp_format)

        file_path = self._write_rgb_png(camera_name, bytes(msg.data))
        if file_path is None:
            return

        self.rgb_captured = True
        self.get_logger().info(f'[{camera_name}] Captured RGB frame: {file_path.name}')
        self._maybe_transition_to_meta(camera_name)

    def _on_depth(self, camera_name: str, msg: CompressedImage) -> None:
        if not self._should_handle(camera_name, 'depth'):
            return
        if self.depth_captured:
            return

        depth_array = self._decode_depth_png(bytes(msg.data))
        if depth_array is None:
            self.get_logger().warning(f'[{camera_name}] Depth decode failed; waiting for next frame.')
            return

        if self.current_timestamp_label is None:
            self.current_timestamp_label = datetime.now().strftime(self.timestamp_format)

        depth_array = np.flipud(depth_array)
        file_path = self._write_depth_png(camera_name, depth_array)
        if file_path is None:
            return

        self.depth_captured = True
        self.get_logger().info(f'[{camera_name}] Captured depth frame: {file_path.name}')
        self._maybe_transition_to_meta(camera_name)

    def _on_meta(self, camera_name: str, msg: CameraInfo) -> None:
        if self._meta_already_recorded(camera_name):
            return
        if not self._should_handle(camera_name, 'meta'):
            return

        self._finalize_meta_capture(camera_name, msg)

    def _should_handle(self, camera_name: str, data_type: str) -> bool:
        if camera_name != self.active_camera.name:
            return False
        if self.stage == 'cooldown':
            return False
        if data_type == 'depth' and not self.current_capture_depth:
            return False
        if data_type in ('rgb', 'depth') and self.stage != 'capture':
            return False
        if data_type == 'meta' and self.stage != 'meta':
            return False
        return True

    def _finalize_meta_capture(self, camera_name: str, msg: CameraInfo) -> None:
        file_path = self._write_meta_file(camera_name, msg)
        if file_path is None:
            self.get_logger().error(f'[{camera_name}] Failed to write camera meta; retrying.')
            return

        self.meta_recorded.add(camera_name)
        self.get_logger().info(
            f'[{camera_name}] Saved meta file: {file_path.name}; starting {self.cooldown_seconds}s pause.'
        )
        self.stage = 'cooldown'
        self._start_cooldown_timer()

    def _handle_meta_stage(self, camera_name: str) -> None:
        if self._meta_already_recorded(camera_name):
            self._skip_meta_capture(camera_name)
        else:
            self.get_logger().info(f'[{camera_name}] Waiting for camera info before saving meta.')

    def _skip_meta_capture(self, camera_name: str) -> None:
        self.stage = 'cooldown'
        self.get_logger().info(
            f'[{camera_name}] Meta already exists; cooling down {self.cooldown_seconds}s.'
        )
        self._start_cooldown_timer()

    def _meta_already_recorded(self, camera_name: str) -> bool:
        if camera_name in self.meta_recorded:
            return True
        file_path = self._meta_file_path(camera_name)
        if file_path.exists():
            self.meta_recorded.add(camera_name)
            return True
        return False

    def _meta_file_path(self, camera_name: str) -> Path:
        return self.meta_dir / f'meta_{camera_name}.mat'

    def _requires_depth_capture(self, rgb_topic: str, depth_topic: str, camera_name: str) -> bool:
        patterns = (rgb_topic or '', depth_topic or '', camera_name or '')
        return any('_car' in token for token in patterns)

    def _reset_capture_state(self) -> None:
        self.stage = 'capture'
        self.current_timestamp_label = None
        self.rgb_captured = False
        self.depth_captured = False

    def _maybe_transition_to_meta(self, camera_name: str) -> None:
        if self.stage != 'capture':
            return
        if self.rgb_captured and self.depth_captured:
            self.stage = 'meta'
            self._handle_meta_stage(camera_name)
        elif not self.rgb_captured:
            self.get_logger().info(f'[{camera_name}] Awaiting RGB frame before meta.')
        elif not self.depth_captured:
            self.get_logger().info(f'[{camera_name}] Awaiting depth frame before meta.')

    def _activate_camera_subscriptions(self, config: CameraConfig) -> None:
        self.get_logger().info(f'Activating camera {config.name} subscriptions.')
        self.current_capture_depth = config.capture_depth
        self._reset_capture_state()
        self._active_subscriptions['rgb'] = self.create_subscription(
            CompressedImage,
            config.rgb_topic,
            partial(self._on_rgb, config.name),
            10,
        )
        if config.capture_depth:
            self._active_subscriptions['depth'] = self.create_subscription(
                CompressedImage,
                config.depth_topic,
                partial(self._on_depth, config.name),
                10,
            )
        else:
            self.depth_captured = True

        self._active_subscriptions['meta'] = self.create_subscription(
            CameraInfo,
            config.meta_topic,
            partial(self._on_meta, config.name),
            10,
        )

    def _deactivate_current_camera(self) -> None:
        if not self._active_subscriptions:
            return
        self.get_logger().info(f'Deactivating camera {self.active_camera.name} subscriptions.')
        for sub in self._active_subscriptions.values():
            self.destroy_subscription(sub)
        self._active_subscriptions.clear()

    def _start_cooldown_timer(self) -> None:
        if self.cooldown_timer is not None:
            self.cooldown_timer.cancel()
            self.destroy_timer(self.cooldown_timer)
            self.cooldown_timer = None

        self.cooldown_timer = self.create_timer(self.cooldown_seconds, self._finish_cooldown)

    def _finish_cooldown(self) -> None:
        if self.cooldown_timer is not None:
            self.cooldown_timer.cancel()
            self.destroy_timer(self.cooldown_timer)
            self.cooldown_timer = None

        self._deactivate_current_camera()
        self.current_camera_index = (self.current_camera_index + 1) % len(self.camera_configs)
        self._activate_camera_subscriptions(self.active_camera)
        self.get_logger().info(f'Switching to next camera: {self.active_camera.name}')

    def _write_rgb_png(self, camera_name: str, data: bytes) -> Optional[Path]:
        filename = f'rgb_{camera_name}_{self.current_timestamp_label}.png'
        file_path = self.rgb_dir / filename
        try:
            with PilImage.open(io.BytesIO(data)) as image:
                image.convert('RGB').save(file_path, format='PNG')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'[{camera_name}] Failed to write RGB image {file_path}: {exc}')
            return None
        return file_path

    def _decode_depth_png(self, data: bytes) -> Optional[np.ndarray]:
        try:
            with PilImage.open(io.BytesIO(data)) as image:
                depth = np.array(image, dtype=np.uint16)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Failed to decode depth PNG: {exc}')
            return None

        if depth.ndim == 3:
            depth = depth[..., 0]

        return depth.astype(np.uint16, copy=False)

    def _write_depth_png(self, camera_name: str, depth: np.ndarray) -> Optional[Path]:
        filename = f'depth_{camera_name}_{self.current_timestamp_label}.png'
        file_path = self.depth_dir / filename
        try:
            PilImage.fromarray(depth, mode='I;16').save(file_path)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'[{camera_name}] Failed to write depth image {file_path}: {exc}')
            return None
        return file_path

    def _write_meta_file(self, camera_name: str, msg: CameraInfo) -> Optional[Path]:
        file_path = self._meta_file_path(camera_name)
        if file_path.exists():
            self.get_logger().info(f'[{camera_name}] Meta file already present at {file_path.name}.')
            return file_path
        payload = self._camera_info_to_payload(camera_name, msg)
        try:
            sio.savemat(file_path, payload)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'[{camera_name}] Failed to save meta MAT file {file_path}: {exc}')
            return None
        return file_path

    def _camera_info_to_payload(self, camera_name: str, msg: CameraInfo) -> Dict[str, np.ndarray]:
        width = float(msg.width)
        height = float(msg.height)

        k_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        fx = k_matrix[0, 0]
        fy = k_matrix[1, 1]
        cx = k_matrix[0, 2]
        cy = k_matrix[1, 2]

        width_arr = np.array([[width]], dtype=np.float64)
        height_arr = np.array([[height]], dtype=np.float64)
        fx_arr = np.array([[fx]], dtype=np.float64)
        fy_arr = np.array([[fy]], dtype=np.float64)
        cx_arr = np.array([[cx]], dtype=np.float64)
        cy_arr = np.array([[cy]], dtype=np.float64)
        near_arr = np.array([[self.near_clip]], dtype=np.float64)
        far_arr = np.array([[self.far_clip]], dtype=np.float64)
        sensor_arr = self.sensor_size_mm.reshape(1, 2)

        # Avoid divide-by-zero when computing derived values.
        focal_length = (
            fx * self.sensor_size_mm[0] / width if width > 0 and fx > 0 else float('nan')
        )
        horizontal_fov = (
            np.degrees(2.0 * np.arctan(width / (2.0 * fx))) if fx > 0 else float('nan')
        )
        vertical_fov = (
            np.degrees(2.0 * np.arctan(height / (2.0 * fy))) if fy > 0 else float('nan')
        )

        focal_arr = np.array([[focal_length]], dtype=np.float64)
        horizontal_arr = np.array([[horizontal_fov]], dtype=np.float64)
        vertical_arr = np.array([[vertical_fov]], dtype=np.float64)

        meta = np.empty((1, 1), dtype=object)
        meta[0, 0] = (
            width_arr,
            height_arr,
            fx_arr,
            fy_arr,
            cx_arr,
            cy_arr,
            k_matrix,
            horizontal_arr,
            vertical_arr,
            focal_arr,
            sensor_arr,
            near_arr,
            far_arr,
        )

        frame_id = msg.header.frame_id if msg.header.frame_id else camera_name

        payload: Dict[str, np.ndarray] = {
            'meta': meta,
            'width': width_arr,
            'height': height_arr,
            'fx': fx_arr,
            'fy': fy_arr,
            'cx': cx_arr,
            'cy': cy_arr,
            'K': k_matrix,
            'intrinsic_matrix': k_matrix.copy(),
            'near': near_arr,
            'far': far_arr,
            'sensorSize': sensor_arr,
            'focalLength': focal_arr,
            'horizontalFov': horizontal_arr,
            'verticalFov': vertical_arr,
            'frame_id': np.array([frame_id], dtype=object),
        }
        return payload


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImagePairSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down compressed image subscriber.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
