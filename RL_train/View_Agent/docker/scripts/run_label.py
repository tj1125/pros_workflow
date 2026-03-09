"""
scripts/run_label.py — Stage 2: Teacher VLM Labeling Entry Point

Reads recorded trajectory HDF5 files and labels them with
Teacher VLM scores (r_total, a_expert, sub-scores).

Usage:
    # Gemini
    TEACHER_VLM_PROVIDER=gemini GEMINI_API_KEY=... \
        python scripts/run_label.py --config configs/train_config.yaml

    # Ollama
    TEACHER_VLM_PROVIDER=ollama OLLAMA_MODEL=gemma3:27b \
        python scripts/run_label.py --config configs/train_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from labeler.teacher_vlm import TeacherVLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="View Agent — Teacher VLM Labeling")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--data-dir", default=None, help="Override data dir")
    parser.add_argument("--skip-labeled", action="store_true",
                        help="Skip trajectories that already have labels")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = Path(args.data_dir or config["labeling"]["data_dir"])
    h5_files = sorted(data_dir.glob("*.h5"))
    logger.info(f"Found {len(h5_files)} trajectory files in {data_dir}")

    teacher = TeacherVLM(config)
    success = 0

    for h5_path in h5_files:
        if args.skip_labeled:
            import h5py
            with h5py.File(h5_path, "r") as f:
                if f.attrs.get("labels_written", False):
                    logger.info(f"Skipping already-labeled: {h5_path.name}")
                    continue

        logger.info(f"Labeling: {h5_path.name}")
        try:
            n = teacher.label_trajectory(h5_path)
            logger.info(f"  → {n} labels written.")
            success += 1
        except Exception as e:
            logger.error(f"  → FAILED: {e}")

    logger.info(f"Labeling complete. {success}/{len(h5_files)} files labeled.")


if __name__ == "__main__":
    main()
