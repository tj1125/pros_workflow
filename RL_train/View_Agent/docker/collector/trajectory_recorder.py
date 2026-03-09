"""
collector/trajectory_recorder.py — Trajectory Data Collector via Rosbridge

Connects to ROS via Rosbridge WebSocket and records synchronized
observation sequences (RGB, Depth, Joints, IMU, Physics Metrics)
into HDF5 files for offline Teacher labeling.

HDF5 Structure per trajectory:
    trajectories/{traj_id}/
        rgb/     (N, H, W, 3) uint8
        depth/   (N, H, W)    float32
        joints/  (N, 6)       float32
        imu/     (N, 6)       float32  [ax,ay,az, wx,wy,wz]
        physics/ (N, 3)       float32  [occlusion, centering, stability]
        actions/ (N, 6)       float32
        metadata             (JSON attr)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import websockets.sync.client as ws_sync
from PIL import Image

from collector.snapshot import PhysicsMetrics, Snapshot

logger = logging.getLogger(__name__)

# Default observation shape after resize
DEFAULT_IMG_SIZE = (224, 224)


class TrajectoryRecorder:
    """
    Subscribes to multiple ROS topics through Rosbridge WebSocket and
    records time-synchronized snapshots at a fixed control rate.

    Usage:
        recorder = TrajectoryRecorder(config)
        recorder.connect()
        traj_id = recorder.record_trajectory(action_fn, duration_sec=8.0)
        recorder.disconnect()
    """

    def __init__(self, config: dict):
        self._url: str = config["env"]["rosbridge_url"]
        self._rgb_topic: str = config["env"]["rgb_topic"]
        self._depth_topic: str = config["env"]["depth_topic"]
        self._joint_topic: str = config["env"]["joint_state_topic"]
        self._imu_topic: str = config["env"]["imu_topic"]
        self._physics_topic: str = config["env"]["physics_metric_topic"]
        self._action_topic: str = config["env"]["action_topic"]
        self._joint_names: List[str] = config["env"]["joint_names"]
        self._out_dir: Path = Path(config["collection"]["out_dir"])
        self._control_hz: float = config["collection"]["control_hz"]
        self._img_size: tuple = tuple(config["collection"]["image_size"])

        # Latest data buffers (updated by message callbacks)
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_joints: Dict[str, float] = {}
        self._latest_imu_lin: List[float] = [0.0] * 3
        self._latest_imu_ang: List[float] = [0.0] * 3
        self._latest_physics: PhysicsMetrics = PhysicsMetrics()

        self._conn = None
        self._out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Open Rosbridge WebSocket connection and subscribe to all topics."""
        self._conn = ws_sync.connect(self._url, open_timeout=10.0)
        for topic, msg_type, cb in [
            (self._rgb_topic,     "sensor_msgs/Image",       self._on_rgb),
            (self._depth_topic,   "sensor_msgs/Image",       self._on_depth),
            (self._joint_topic,   "sensor_msgs/JointState",  self._on_joints),
            (self._imu_topic,     "sensor_msgs/Imu",         self._on_imu),
            (self._physics_topic, "std_msgs/String",         self._on_physics),
        ]:
            self._subscribe(topic, msg_type, cb)
        logger.info(f"[Recorder] Connected to Rosbridge at {self._url}")

    def disconnect(self) -> None:
        """Close the Rosbridge connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("[Recorder] Disconnected from Rosbridge.")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record_trajectory(
        self,
        action_fn,
        duration_sec: float = 8.0,
        env_seed: Optional[int] = None,
    ) -> str:
        """
        Record one trajectory of observations + actions into HDF5.

        Args:
            action_fn   : Callable[[], List[float]] — returns 6-DOF action each step.
            duration_sec: Wall-clock duration of the trajectory.
            env_seed    : Random seed used for Unity environment (for metadata).

        Returns:
            traj_id (str): The UUID of the saved trajectory.
        """
        traj_id = uuid.uuid4().hex
        control_period = 1.0 / self._control_hz
        n_steps = int(duration_sec * self._control_hz)

        snapshots: List[Snapshot] = []

        logger.info(
            f"[Recorder] Recording trajectory {traj_id} "
            f"({n_steps} steps @ {self._control_hz} Hz)"
        )

        for step_idx in range(n_steps):
            t0 = time.monotonic()

            # Spin Rosbridge to receive pending messages
            self._spin(timeout=0.01)

            # Sample action from policy
            action = action_fn()

            # Publish action to Unity
            self._publish_action(action)

            # Capture synchronized snapshot
            snap = self._capture_snapshot(step_idx, action)
            snapshots.append(snap)

            # Rate control
            elapsed = time.monotonic() - t0
            sleep_time = control_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Save to HDF5
        out_path = self._out_dir / f"{traj_id}.h5"
        self._save_hdf5(out_path, traj_id, snapshots, env_seed=env_seed)
        logger.info(f"[Recorder] Saved {len(snapshots)} steps → {out_path}")
        return traj_id

    # ------------------------------------------------------------------
    # Snapshot capture
    # ------------------------------------------------------------------
    def _capture_snapshot(self, step_idx: int, action: list) -> Snapshot:
        """Build a Snapshot from the latest buffered sensor data."""
        rgb_bytes = b""
        depth_bytes = b""

        if self._latest_rgb is not None:
            img_pil = Image.fromarray(self._latest_rgb).resize(self._img_size)
            buf = BytesIO()
            img_pil.save(buf, format="PNG")
            rgb_bytes = buf.getvalue()

        if self._latest_depth is not None:
            # Store depth as float32 PNG (scale to uint16 for lossless PNG)
            depth_norm = np.clip(self._latest_depth, 0, 10.0)  # clip at 10m
            depth_u16 = (depth_norm / 10.0 * 65535).astype(np.uint16)
            depth_pil = Image.fromarray(depth_u16)
            buf = BytesIO()
            depth_pil.save(buf, format="PNG")
            depth_bytes = buf.getvalue()

        return Snapshot(
            step_idx=step_idx,
            timestamp=time.time(),
            rgb_bytes=rgb_bytes,
            depth_bytes=depth_bytes,
            joint_angles=dict(self._latest_joints),
            linear_acceleration=list(self._latest_imu_lin),
            angular_velocity=list(self._latest_imu_ang),
            physics=PhysicsMetrics(
                occlusion_rate=self._latest_physics.occlusion_rate,
                centering_score=self._latest_physics.centering_score,
                chassis_stability=self._latest_physics.chassis_stability,
            ),
            action=list(action),
        )

    # ------------------------------------------------------------------
    # HDF5 persistence
    # ------------------------------------------------------------------
    @staticmethod
    def _save_hdf5(
        path: Path,
        traj_id: str,
        snapshots: List[Snapshot],
        env_seed: Optional[int] = None,
    ) -> None:
        """Save a list of Snapshots to an HDF5 file."""
        n = len(snapshots)
        if n == 0:
            logger.warning("[Recorder] No snapshots to save.")
            return

        with h5py.File(path, "w") as f:
            # Variable-length bytes datasets for images
            vlen_bytes = h5py.special_dtype(vlen=bytes)
            rgb_ds = f.create_dataset("rgb", (n,), dtype=vlen_bytes)
            depth_ds = f.create_dataset("depth", (n,), dtype=vlen_bytes)

            # Fixed-shape numeric datasets
            joints_ds = f.create_dataset("joints", (n, 6), dtype=np.float32)
            imu_ds = f.create_dataset("imu", (n, 6), dtype=np.float32)
            physics_ds = f.create_dataset("physics", (n, 3), dtype=np.float32)
            actions_ds = f.create_dataset("actions", (n, 6), dtype=np.float32)
            timestamps_ds = f.create_dataset("timestamps", (n,), dtype=np.float64)

            for i, snap in enumerate(snapshots):
                rgb_ds[i] = snap.rgb_bytes
                depth_ds[i] = snap.depth_bytes
                joints_arr = [
                    snap.joint_angles.get(j, 0.0)
                    for j in (list(snap.joint_angles.keys())[:6])
                ]
                joints_arr += [0.0] * (6 - len(joints_arr))
                joints_ds[i] = joints_arr
                imu_ds[i] = snap.linear_acceleration + snap.angular_velocity
                physics_ds[i] = [
                    snap.physics.occlusion_rate,
                    snap.physics.centering_score,
                    snap.physics.chassis_stability,
                ]
                actions_ds[i] = snap.action
                timestamps_ds[i] = snap.timestamp

            # Metadata as JSON attribute
            f.attrs["traj_id"] = traj_id
            f.attrs["n_steps"] = n
            f.attrs["env_seed"] = env_seed or -1
            f.attrs["timestamp_start"] = snapshots[0].timestamp
            f.attrs["labels_written"] = False  # Will be set True by label_writer

    # ------------------------------------------------------------------
    # Rosbridge helpers
    # ------------------------------------------------------------------
    def _subscribe(self, topic: str, msg_type: str, callback) -> None:
        cmd = {
            "op": "subscribe",
            "topic": topic,
            "type": msg_type,
            "queue_length": 1,
            "throttle_rate": 0,
        }
        self._conn.send(json.dumps(cmd))

    def _spin(self, timeout: float = 0.01) -> None:
        """Non-blocking receive: drain available messages from Rosbridge."""
        import select, socket
        sock = self._conn.socket
        try:
            # Try to receive up to 10 messages without blocking too long
            for _ in range(10):
                ready = select.select([sock], [], [], timeout)
                if not ready[0]:
                    break
                raw = self._conn.recv()
                msg = json.loads(raw)
                if msg.get("op") == "publish":
                    self._dispatch(msg)
        except Exception:
            pass

    def _dispatch(self, msg: dict) -> None:
        topic = msg.get("topic", "")
        data = msg.get("msg", {})
        if topic == self._rgb_topic:
            self._on_rgb(data)
        elif topic == self._depth_topic:
            self._on_depth(data)
        elif topic == self._joint_topic:
            self._on_joints(data)
        elif topic == self._imu_topic:
            self._on_imu(data)
        elif topic == self._physics_topic:
            self._on_physics(data)

    def _publish_action(self, action: list) -> None:
        """Publish 6-DOF delta joint action to Unity."""
        cmd = {
            "op": "publish",
            "topic": self._action_topic,
            "msg": {"data": json.dumps(action)},
        }
        try:
            self._conn.send(json.dumps(cmd))
        except Exception as e:
            logger.warning(f"[Recorder] Failed to publish action: {e}")

    # ------------------------------------------------------------------
    # ROS message callbacks
    # ------------------------------------------------------------------
    def _on_rgb(self, msg: dict) -> None:
        """Decode sensor_msgs/Image to numpy array."""
        try:
            import base64
            raw = base64.b64decode(msg["data"])
            arr = np.frombuffer(raw, dtype=np.uint8)
            h, w = msg["height"], msg["width"]
            self._latest_rgb = arr.reshape(h, w, -1)[..., :3]
        except Exception:
            pass

    def _on_depth(self, msg: dict) -> None:
        """Decode depth image to float32 numpy array."""
        try:
            import base64
            raw = base64.b64decode(msg["data"])
            arr = np.frombuffer(raw, dtype=np.uint16)
            h, w = msg["height"], msg["width"]
            self._latest_depth = arr.reshape(h, w).astype(np.float32) / 1000.0  # mm→m
        except Exception:
            pass

    def _on_joints(self, msg: dict) -> None:
        """Update joint angles from JointState message."""
        try:
            names = msg.get("name", [])
            positions = msg.get("position", [])
            self._latest_joints = dict(zip(names, positions))
        except Exception:
            pass

    def _on_imu(self, msg: dict) -> None:
        """Update IMU buffers from Imu message."""
        try:
            la = msg["linear_acceleration"]
            av = msg["angular_velocity"]
            self._latest_imu_lin = [la["x"], la["y"], la["z"]]
            self._latest_imu_ang = [av["x"], av["y"], av["z"]]
        except Exception:
            pass

    def _on_physics(self, msg: dict) -> None:
        """Update physics metrics from JSON string message."""
        try:
            data = json.loads(msg["data"])
            self._latest_physics = PhysicsMetrics(
                occlusion_rate=float(data.get("occlusion_rate", 0.0)),
                centering_score=float(data.get("centering_score", 0.0)),
                chassis_stability=float(data.get("chassis_stability", 1.0)),
            )
        except Exception:
            pass
