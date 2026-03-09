"""
labeler/teacher_vlm.py — Teacher VLM Batch Labeling

Reads recorded trajectory HDF5 files, sends observation windows
to a large VLM (Gemini or Ollama), and writes back labels via
LabelWriter.

Supports:
  - Google Gemini (via google-generativeai)
  - Ollama (via httpx REST API)

Environment variables:
  TEACHER_VLM_PROVIDER  : "gemini" | "ollama"   (default: "gemini")
  GEMINI_API_KEY        : Gemini API key
  OLLAMA_BASE_URL       : Ollama server URL      (default: http://localhost:11434)
  OLLAMA_MODEL          : Ollama model name      (default: "gemma3:27b")
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
from PIL import Image

from labeler.label_writer import LabelResult, LabelWriter
from labeler.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class TeacherVLM:
    """
    Teacher VLM: reads HDF5 trajectories, sends sliding-window frames
    to a large VLM for temporal evaluation, and writes labels back.

    Usage:
        teacher = TeacherVLM(config)
        teacher.label_trajectory(hdf5_path)
    """

    def __init__(self, config: dict):
        self._provider: str = os.getenv(
            "TEACHER_VLM_PROVIDER",
            config["labeling"].get("provider", "gemini")
        ).lower()
        self._gemini_model: str = config["labeling"].get("gemini_model", "gemini-1.5-pro")
        self._ollama_model: str = os.getenv(
            "OLLAMA_MODEL",
            config["labeling"].get("ollama_model", "gemma3:27b")
        )
        self._ollama_url: str = os.getenv(
            "OLLAMA_BASE_URL",
            config["labeling"].get("ollama_base_url", "http://localhost:11434")
        )
        self._window_size: int = config["labeling"].get("window_size", 4)
        self._reward_weights: dict = config["labeling"].get("reward_weights", {})
        self._prompt_builder = PromptBuilder()

        # Lazy-initialized VLM client
        self._gemini_client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def label_trajectory(self, hdf5_path: Path) -> int:
        """
        Label one trajectory file.

        Args:
            hdf5_path: Path to an existing .h5 trajectory file.

        Returns:
            Number of steps successfully labeled.
        """
        labels: List[LabelResult] = []

        with h5py.File(hdf5_path, "r") as f:
            n_steps = int(f.attrs.get("n_steps", f["actions"].shape[0]))
            rgb_ds = f["rgb"]
            physics_ds = f["physics"][:]   # (N, 3) float32
            actions_ds = f["actions"][:]   # (N, 6) float32

        # Slide window across trajectory
        for end_idx in range(self._window_size - 1, n_steps):
            start_idx = end_idx - self._window_size + 1

            # Build frame data list for Physics metrics
            frames_data: List[Dict] = []
            base64_imgs: List[str] = []

            with h5py.File(hdf5_path, "r") as f:
                for i in range(start_idx, end_idx + 1):
                    phys = physics_ds[i]
                    frames_data.append({
                        "occlusion_rate":   float(phys[0]),
                        "centering_score":  float(phys[1]),
                        "chassis_stability": float(phys[2]),
                        "action":           actions_ds[i].tolist(),
                    })
                    # Decode RGB image for VLM
                    rgb_bytes = bytes(f["rgb"][i])
                    if rgb_bytes:
                        img = Image.open(BytesIO(rgb_bytes)).convert("RGB")
                        buf = BytesIO()
                        img.save(buf, format="JPEG", quality=85)
                        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        base64_imgs.append(b64)
                    else:
                        base64_imgs.append("")

            # Query Teacher VLM
            system_p, user_p, imgs = self._prompt_builder.build(frames_data, base64_imgs)
            try:
                response_dict = self._query_vlm(system_p, user_p, imgs)
                label = LabelResult.from_vlm_response(end_idx, response_dict)
                labels.append(label)
                logger.debug(
                    f"[Teacher] Step {end_idx}: r_total={label.r_total:.3f} "
                    f"| {label.reasoning}"
                )
            except Exception as e:
                logger.warning(f"[Teacher] Step {end_idx} labeling failed: {e}")

            # Small delay to respect API rate limits
            time.sleep(0.05)

        # Write labels back to HDF5
        LabelWriter.write(hdf5_path, labels)
        logger.info(
            f"[Teacher] Labeled {len(labels)}/{n_steps} windows in {hdf5_path.name}"
        )
        return len(labels)

    # ------------------------------------------------------------------
    # VLM query dispatch
    # ------------------------------------------------------------------
    def _query_vlm(
        self, system_prompt: str, user_text: str, base64_images: List[str]
    ) -> dict:
        """Route query to Gemini or Ollama and return parsed JSON dict."""
        if self._provider == "gemini":
            return self._query_gemini(system_prompt, user_text, base64_images)
        elif self._provider == "ollama":
            return self._query_ollama(system_prompt, user_text, base64_images)
        else:
            raise ValueError(f"Unknown VLM provider: {self._provider}")

    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------
    def _query_gemini(
        self, system_prompt: str, user_text: str, base64_images: List[str]
    ) -> dict:
        """Query Google Gemini VLM."""
        import google.generativeai as genai

        if self._gemini_client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set.")
            genai.configure(api_key=api_key)
            self._gemini_client = genai.GenerativeModel(
                model_name=self._gemini_model,
                system_instruction=system_prompt,
            )

        # Build content parts: alternating image + text
        parts = []
        for b64 in base64_images:
            if b64:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64,
                    }
                })
        parts.append({"text": user_text})

        response = self._gemini_client.generate_content(parts)
        return self._parse_json_response(response.text)

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------
    def _query_ollama(
        self, system_prompt: str, user_text: str, base64_images: List[str]
    ) -> dict:
        """Query Ollama VLM via REST API."""
        import httpx

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_text,
                "images": [b for b in base64_images if b],
            },
        ]

        payload = {
            "model": self._ollama_model,
            "messages": messages,
            "stream": False,
            "format": "json",
        }

        resp = httpx.post(
            f"{self._ollama_url}/api/chat",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return self._parse_json_response(content)

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """
        Extract and parse the JSON block from VLM response text.
        Handles cases where VLM wraps JSON in markdown code fences.
        """
        text = text.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Find the first { ... } block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON found in VLM response: {text[:200]}")

        return json.loads(text[start:end])
