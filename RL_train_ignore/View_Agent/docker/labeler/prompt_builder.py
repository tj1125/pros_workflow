"""
labeler/prompt_builder.py — Teacher VLM Prompt Builder

Builds structured prompts for the Teacher VLM to evaluate a
sequence of observation frames and provide:
  1. A composite reward score r_total ∈ [-1, 1]
  2. A recommended 6-DOF arm delta action a_expert
  3. Per-dimension sub-scores for analysis

The prompt follows a "Chain-of-Thought" format so the large VLM
reasons step by step before giving the final score.
"""

from __future__ import annotations

from typing import Dict, List


SYSTEM_PROMPT = """You are an expert robot manipulation teacher evaluating a robot arm's 
viewpoint adjustment behavior. The robot must adjust its arm pose to improve visual clarity 
of a target object that is partially occluded by obstacles or the gripper itself.

You will receive a sequence of observation frames (oldest to newest) with physical metrics 
for each frame. Analyze how the robot's view of the target object changes over time.

Your evaluation criteria:
1. OCCLUSION IMPROVEMENT: Did occlusion_rate decrease from earlier to later frames? 
   Lower occlusion is better.
2. CENTERING: Is the target object moving toward the camera center (higher centering_score)?
3. STABILITY: Is the robot chassis stable (high chassis_stability = good)?
4. PROGRESS: Compare the LATEST frame to EARLIER frames — is the situation improving?
5. CONSISTENCY: Are the actions logical and coherent, or random/oscillating?
6. SEMANTIC CONTEXT: Any special situations (glare, gripper blocking key grasp point, etc.)?

Output ONLY valid JSON, no markdown, no explanation outside the JSON.
"""

EVAL_PROMPT_TEMPLATE = """Evaluate this {n_frames}-frame observation sequence for the View Agent.

=== PHYSICAL METRICS (per frame, index 0=oldest, {last_idx}=newest) ===
{metrics_table}

=== TASK ===
The robot must adjust its arm to reduce occlusion_rate and improve centering_score 
while keeping chassis_stability high.

=== YOUR EVALUATION ===
Think step by step:
1. Analyze the trend in occlusion_rate across frames.
2. Analyze centering_score trend.
3. Judge action consistency (logical progression vs random oscillation).
4. Identify any semantic issues from the metrics.
5. Assign sub-scores and compute r_total.

Output this JSON (all numbers are float, action is a list of 6 floats in radians):
{{
  "occlusion_improvement": <float in [-1,1]>,
  "centering_score":       <float in [0,1]>,
  "chassis_stability":     <float in [0,1]>,
  "progress_reward":       <float in [-1,1]>,
  "consistency":           <float in [0,1]>,
  "semantic_context":      <float in [0,1]>,
  "r_total":               <float in [-1,1]>,
  "a_expert":              [j1, j2, j3, j4, j5, j6],
  "reasoning":             "<one sentence explanation>"
}}"""


class PromptBuilder:
    """
    Builds Teacher VLM evaluation prompts from a window of Snapshot data.

    Usage:
        builder = PromptBuilder()
        system, user = builder.build(frames_data)
    """

    def build(
        self,
        frames_data: List[Dict],
        base64_images: List[str],
    ) -> tuple[str, str, List[str]]:
        """
        Build system prompt, user text prompt, and image list for the Teacher VLM.

        Args:
            frames_data   : List of per-frame metric dicts (from HDF5):
                            {occlusion_rate, centering_score, chassis_stability, action}
            base64_images : List of base64-encoded RGB images (same length as frames_data)

        Returns:
            (system_prompt, user_text_prompt, base64_images)
        """
        n = len(frames_data)
        metrics_table = self._build_metrics_table(frames_data)

        user_text = EVAL_PROMPT_TEMPLATE.format(
            n_frames=n,
            last_idx=n - 1,
            metrics_table=metrics_table,
        )

        return SYSTEM_PROMPT.strip(), user_text, base64_images

    @staticmethod
    def _build_metrics_table(frames_data: List[Dict]) -> str:
        """Format per-frame metrics as a readable table."""
        lines = [
            f"{'Frame':>5} | {'Occlusion':>10} | {'Centering':>9} | "
            f"{'Stability':>9} | {'Action (rad, 6-DOF)'}",
            "-" * 80,
        ]
        for i, fd in enumerate(frames_data):
            action_str = "[" + ", ".join(f"{a:+.3f}" for a in fd.get("action", [0]*6)) + "]"
            lines.append(
                f"{i:>5} | {fd.get('occlusion_rate', 0):>10.3f} | "
                f"{fd.get('centering_score', 0):>9.3f} | "
                f"{fd.get('chassis_stability', 1):>9.3f} | {action_str}"
            )
        return "\n".join(lines)
