"""
collector/randomizer.py — Unity Environment Randomizer

Sends randomization commands to Unity via Rosbridge WebSocket,
triggering random placement of obstacles, target objects, lighting,
and chassis slope for training diversity.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Optional

import websockets.sync.client as ws_sync

logger = logging.getLogger(__name__)


class EnvRandomizer:
    """
    Sends a '/env_randomize' ROS topic message to Unity through Rosbridge.

    The Unity-side EnvRandomizer.cs subscribes to this topic and applies
    the randomization parameters.

    Usage:
        randomizer = EnvRandomizer("ws://localhost:9090")
        randomizer.randomize()
    """

    # ROS topic Unity listens to for randomization commands
    TOPIC = "/env_randomize"
    MSG_TYPE = "std_msgs/String"

    def __init__(self, rosbridge_url: str):
        self._url = rosbridge_url

    def randomize(self, seed: Optional[int] = None) -> dict:
        """
        Send a randomization request to Unity.

        Args:
            seed: Optional fixed seed for reproducibility.
                  If None, a random seed is generated.

        Returns:
            The randomization parameters dict that was sent.
        """
        seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        params = self._build_params(seed)

        self._publish(json.dumps(params))
        logger.info(f"[Randomizer] Sent randomization with seed={seed}")
        return params

    def wait_for_ready(self, timeout_sec: float = 5.0) -> bool:
        """
        Block until Unity acknowledges reset completion.
        Currently implemented as a fixed delay; can be upgraded to
        a service call or status topic in the future.
        """
        time.sleep(min(timeout_sec, 2.0))
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_params(seed: int) -> dict:
        """Generate randomization parameters from seed."""
        rng = random.Random(seed)
        return {
            "seed": seed,
            # Obstacle count and positions
            "num_obstacles": rng.randint(1, 5),
            # Target object type index
            "target_object_idx": rng.randint(0, 9),
            # Lighting: [0=normal, 1=bright, 2=dim, 3=backlit]
            "lighting_mode": rng.randint(0, 3),
            # Chassis slope angle (degrees)
            "chassis_slope_deg": rng.uniform(-10.0, 10.0),
            # Camera noise level [0=clean, 1=noisy]
            "camera_noise": rng.uniform(0.0, 0.2),
        }

    def _publish(self, message: str) -> None:
        """Publish a single message to Rosbridge and close immediately."""
        advertise_cmd = {
            "op": "advertise",
            "topic": self.TOPIC,
            "type": self.MSG_TYPE,
        }
        publish_cmd = {
            "op": "publish",
            "topic": self.TOPIC,
            "msg": {"data": message},
        }
        unadvertise_cmd = {
            "op": "unadvertise",
            "topic": self.TOPIC,
        }

        try:
            with ws_sync.connect(self._url) as conn:
                conn.send(json.dumps(advertise_cmd))
                conn.send(json.dumps(publish_cmd))
                conn.send(json.dumps(unadvertise_cmd))
        except Exception as e:
            logger.error(f"[Randomizer] Failed to publish to Rosbridge: {e}")
