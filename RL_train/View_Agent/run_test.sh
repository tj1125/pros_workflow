#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR="${1:-$SCRIPT_DIR/test_tmp/pics/Camera_Room1_test}"
TARGET_LABEL="${2:-doll}"
CAMERA_IDS=(Camera_Room1_2 Camera_Room1_5 Camera_Room1_7 Camera_Room1_10)

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  if __conda_setup="$(conda shell.bash hook 2>/dev/null)"; then
    eval "$__conda_setup"
  fi
fi

if command -v conda >/dev/null 2>&1; then
  if [[ "${CONDA_DEFAULT_ENV:-}" != "a2a_vlm_find" ]]; then
    conda activate a2a_vlm_find
  fi
fi

echo "Running point-based height test..."
python "$SCRIPT_DIR/test_multicam_teddy_height.py" \
  --image-dir "$IMAGE_DIR" \
  --camera-ids "${CAMERA_IDS[@]}" \
  --target-labels "$TARGET_LABEL" \
  --min-views 4

echo
echo "Running bbox volume carving test..."
python "$SCRIPT_DIR/test_multicam_bbox_volume_height.py" \
  --image-dir "$IMAGE_DIR" \
  --camera-ids "${CAMERA_IDS[@]}" \
  --target-labels "$TARGET_LABEL" \
  --min-views 4

echo
echo "Running mask visual hull test..."
python "$SCRIPT_DIR/test_multicam_mask_visual_hull_height.py" \
  --image-dir "$IMAGE_DIR" \
  --camera-ids "${CAMERA_IDS[@]}" \
  --target-labels "$TARGET_LABEL" \
  --min-views 4

echo
python - "$IMAGE_DIR" "$TARGET_LABEL" <<'PY'
import json
import sys
from pathlib import Path


def normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


def tag(label: str) -> str:
    return normalize_label(label).replace(" ", "_")


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


image_dir = Path(sys.argv[1]).resolve()
target_label = sys.argv[2]
target_tag = tag(target_label)

point_report = image_dir / f"{target_tag}_height_outputs" / f"multicam_{target_tag}_height_report.json"
bbox_report = image_dir / f"{target_tag}_bbox_volume_outputs" / f"multicam_{target_tag}_bbox_volume_report.json"
mask_report = image_dir / f"{target_tag}_mask_visual_hull_outputs" / f"multicam_{target_tag}_mask_visual_hull_report.json"

reports = {
    "Point Triangulation": json.loads(point_report.read_text()),
    "BBox Volume": json.loads(bbox_report.read_text()),
    "Mask Visual Hull": json.loads(mask_report.read_text()),
}

rows = [
    {
        "Method": "Point Triangulation",
        "Height_mm": fmt(reports["Point Triangulation"].get("estimated_height_mm")),
        "Cameras": ",".join(reports["Point Triangulation"].get("selected_cameras", [])),
        "Reproj_px": fmt(reports["Point Triangulation"].get("reprojection_error_px")),
        "Volume_cm3": "-",
        "Voxels": "-",
        "Notes": f"offset_mm={fmt(reports['Point Triangulation'].get('height_estimates_m', {}).get('height_offset_mm'), 1)}",
    },
    {
        "Method": "BBox Volume",
        "Height_mm": fmt(reports["BBox Volume"].get("volume_height_mm")),
        "Cameras": ",".join(reports["BBox Volume"].get("selected_cameras", [])),
        "Reproj_px": fmt(reports["BBox Volume"].get("reprojection_error_px")),
        "Volume_cm3": fmt(reports["BBox Volume"].get("volume_m3", 0.0) * 1e6),
        "Voxels": str(reports["BBox Volume"].get("occupied_voxel_count", "-")),
        "Notes": f"bbox_pad_px={fmt(reports['BBox Volume'].get('bbox_padding_px'), 1)}",
    },
    {
        "Method": "Mask Visual Hull",
        "Height_mm": fmt(reports["Mask Visual Hull"].get("volume_height_mm")),
        "Cameras": ",".join(reports["Mask Visual Hull"].get("selected_cameras", [])),
        "Reproj_px": fmt(reports["Mask Visual Hull"].get("reprojection_error_px")),
        "Volume_cm3": fmt(reports["Mask Visual Hull"].get("volume_m3", 0.0) * 1e6),
        "Voxels": str(reports["Mask Visual Hull"].get("occupied_voxel_count", "-")),
        "Notes": (
            f"mask={reports['Mask Visual Hull'].get('mask_method', '-')}, "
            f"dilate_px={fmt(reports['Mask Visual Hull'].get('mask_dilate_px'), 0)}"
        ),
    },
]

headers = ["Method", "Height_mm", "Cameras", "Reproj_px", "Volume_cm3", "Voxels", "Notes"]
table_lines = [
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]
for row in rows:
    table_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

report_lines = [
    f"# Multicam Test Summary: {target_label}",
    "",
    f"Image dir: `{image_dir}`",
    "",
    *table_lines,
    "",
    "## Report Paths",
    f"- Point Triangulation: `{point_report}`",
    f"- BBox Volume: `{bbox_report}`",
    f"- Mask Visual Hull: `{mask_report}`",
]

summary_path = image_dir / f"{target_tag}_test_summary.md"
summary_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
print("\n".join(report_lines))
print()
print(f"Summary saved to: {summary_path}")
PY
