"""
labeler/label_writer.py — Write Teacher VLM Labels Back to HDF5

Opens existing trajectory HDF5 files and writes the Teacher's
evaluation results (r_total, a_expert, sub-scores) into a
'labels/' group within the same file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LabelResult:
    """
    Structured result from one Teacher VLM evaluation window.
    Covers a window of frames at step_idx (the last frame in the window).
    """
    step_idx: int                           # Last frame index in the window
    r_total: float = 0.0                    # Composite reward ∈ [-1, 1]
    a_expert: List[float] = field(default_factory=lambda: [0.0] * 6)  # Expert action (rad)

    # Sub-scores from Teacher
    occlusion_improvement: float = 0.0
    centering_score: float = 0.0
    chassis_stability: float = 1.0
    progress_reward: float = 0.0
    consistency: float = 0.0
    semantic_context: float = 0.0

    reasoning: str = ""                     # One-sentence Teacher reasoning

    @classmethod
    def from_vlm_response(cls, step_idx: int, response: dict) -> "LabelResult":
        """Parse a LabelResult from the Teacher VLM's JSON response dict."""
        return cls(
            step_idx=step_idx,
            r_total=float(response.get("r_total", 0.0)),
            a_expert=[float(x) for x in response.get("a_expert", [0.0] * 6)],
            occlusion_improvement=float(response.get("occlusion_improvement", 0.0)),
            centering_score=float(response.get("centering_score", 0.0)),
            chassis_stability=float(response.get("chassis_stability", 1.0)),
            progress_reward=float(response.get("progress_reward", 0.0)),
            consistency=float(response.get("consistency", 0.0)),
            semantic_context=float(response.get("semantic_context", 0.0)),
            reasoning=str(response.get("reasoning", "")),
        )


class LabelWriter:
    """
    Writes Teacher VLM labels into the HDF5 trajectory file under a 'labels/' group.

    HDF5 schema added:
        labels/r_total         (N,)   float32
        labels/a_expert        (N, 6) float32
        labels/occlusion_imp   (N,)   float32
        labels/centering       (N,)   float32
        labels/stability       (N,)   float32
        labels/progress        (N,)   float32
        labels/consistency     (N,)   float32
        labels/semantic        (N,)   float32
        labels/reasoning       (N,)   variable-length string

    Steps without a Teacher label (e.g., early frames before the window fills)
    are filled with NaN.
    """

    @staticmethod
    def write(hdf5_path: Path, labels: List[LabelResult]) -> None:
        """
        Write a list of LabelResults into an existing HDF5 trajectory file.

        Args:
            hdf5_path: Path to the .h5 trajectory file.
            labels   : List of LabelResult objects.
        """
        with h5py.File(hdf5_path, "a") as f:
            n_steps = int(f.attrs.get("n_steps", len(f["actions"])))

            # Remove existing labels group if present (allow re-labeling)
            if "labels" in f:
                del f["labels"]
            grp = f.create_group("labels")

            # Initialize arrays with NaN
            r_total = np.full(n_steps, np.nan, dtype=np.float32)
            a_expert = np.full((n_steps, 6), np.nan, dtype=np.float32)
            occ_imp = np.full(n_steps, np.nan, dtype=np.float32)
            centering = np.full(n_steps, np.nan, dtype=np.float32)
            stability = np.full(n_steps, np.nan, dtype=np.float32)
            progress = np.full(n_steps, np.nan, dtype=np.float32)
            consistency = np.full(n_steps, np.nan, dtype=np.float32)
            semantic = np.full(n_steps, np.nan, dtype=np.float32)
            reasoning_list = [""] * n_steps

            for lbl in labels:
                idx = lbl.step_idx
                if 0 <= idx < n_steps:
                    r_total[idx] = lbl.r_total
                    a_expert[idx] = lbl.a_expert[:6]
                    occ_imp[idx] = lbl.occlusion_improvement
                    centering[idx] = lbl.centering_score
                    stability[idx] = lbl.chassis_stability
                    progress[idx] = lbl.progress_reward
                    consistency[idx] = lbl.consistency
                    semantic[idx] = lbl.semantic_context
                    reasoning_list[idx] = lbl.reasoning

            grp.create_dataset("r_total",       data=r_total)
            grp.create_dataset("a_expert",       data=a_expert)
            grp.create_dataset("occlusion_imp",  data=occ_imp)
            grp.create_dataset("centering",      data=centering)
            grp.create_dataset("stability",      data=stability)
            grp.create_dataset("progress",       data=progress)
            grp.create_dataset("consistency",    data=consistency)
            grp.create_dataset("semantic",       data=semantic)

            # Variable-length string dataset
            vlen_str = h5py.special_dtype(vlen=str)
            reasoning_ds = grp.create_dataset("reasoning", (n_steps,), dtype=vlen_str)
            for i, r in enumerate(reasoning_list):
                reasoning_ds[i] = r

            # Mark trajectory as labeled
            f.attrs["labels_written"] = True
            f.attrs["n_labels"] = len(labels)

        logger.info(
            f"[LabelWriter] Wrote {len(labels)} labels → {hdf5_path.name}"
        )
