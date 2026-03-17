from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


class Sam3DInference:
    """Minimal SAM3D runtime adapter (no notebook/UI dependencies)."""

    def __init__(self, config_file: Path, compile_model: bool = False, device: str | None = None):
        try:
            # Fail early with a clear message instead of Hydra's generic target error.
            from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Failed to import SAM3D runtime dependencies. "
                f"Root cause: {type(exc).__name__}: {exc}. "
                "Install missing packages (commonly: optree, astor, easydict) and retry."
            ) from exc

        config_path = Path(config_file).resolve()
        cfg = OmegaConf.load(str(config_path))
        resolved_device = str(device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("SAM3D requested CUDA but torch.cuda.is_available() is false.")
        cfg.rendering_engine = "pytorch3d"
        cfg.compile_model = compile_model
        cfg.workspace_dir = str(config_path.parent)
        cfg.device = resolved_device
        if "depth_model" in cfg:
            try:
                cfg.depth_model.device = resolved_device
            except Exception:
                pass

        # Avoid side effects from optional runtime initialization.
        os.environ.setdefault("LIDRA_SKIP_INIT", "true")
        self._pipeline = instantiate(cfg)
        actual_device = str(getattr(self._pipeline, "device", resolved_device))
        if resolved_device.startswith("cuda") and not actual_device.startswith("cuda"):
            raise RuntimeError(
                f"SAM3D instantiated on {actual_device} instead of requested {resolved_device}."
            )

    @staticmethod
    def _merge_mask_to_rgba(image_rgb: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        """Merge an RGB image and a boolean mask into an RGBA array."""
        alpha = (mask_bool.astype(np.uint8) * 255)[..., None]
        return np.concatenate([image_rgb[..., :3], alpha], axis=-1)

    def run(self, image_rgb: np.ndarray, mask_bool: np.ndarray, seed: int) -> dict[str, Any]:
        """Run SAM3D inference and return the output dict (contains 'glb' mesh)."""
        rgba = self._merge_mask_to_rgba(image_rgb, mask_bool)
        return self._pipeline.run(
            rgba,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=True,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=None,
        )
