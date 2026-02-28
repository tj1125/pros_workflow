import asyncio
import logging
import subprocess
import os
from typing import Optional

logger = logging.getLogger(__name__)

async def get_camera_image_base64(camera_name: str = "Camera_Car", timeout_sec: float = 10.0) -> Optional[str]:
    """
    Run a separate Python 3.10 process to invoke rclpy (since our venv uses Python 3.12).
    This cleanly avoids C-extension version conflicts (rclpy compiled for 3.10 but running in 3.12).
    """
    loop = asyncio.get_event_loop()
    
    def _capture():
        script_path = os.path.join(os.path.dirname(__file__), "camera_310.py")
        
        logger.info(f"[Camera] Requesting image from Unity ({camera_name}) via Python 3.10 subprocess...")
        try:
            # We explicitly call the system python which has ROS 2 packages installed
            result = subprocess.run(
                ["/usr/bin/python3", script_path, camera_name],
                capture_output=True,
                text=True,
                timeout=timeout_sec + 2.0
            )
            
            if result.returncode == 0 and result.stdout.strip():
                logger.info("[Camera] Image received successfully.")
                return result.stdout.strip()
            else:
                logger.error(f"[Camera] Failed to capture image. Error Output: {result.stderr.strip()}")
                return None
        except Exception as e:
            logger.error(f"[Camera] Subprocess exception: {e}")
            return None
            
    return await loop.run_in_executor(None, _capture)
