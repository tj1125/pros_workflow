#!/usr/bin/env bash
# setup_nv.sh — one command to build the NVIDIA CUDA conda env for the perception +
# grasp services (NVIDIA CUDA 12.1). The CUDA counterpart of setup_rocm.sh.
#
# Usage (from this folder):
#   bash setup_nv.sh
# Optional overrides:
#   ENV_NAME=pros_workflow PYVER=3.11 CUDA_TAG=cu121 bash setup_nv.sh
#
# It: creates the conda env, installs the CUDA PyTorch wheels, installs the full
# perception/GraspGen deps (requirements-nv.txt — includes the SAM3D stack, so the
# legacy :8008 service also works on CUDA), builds the GraspGen pointnet2 kernel with
# nvcc, and verifies the GPU works.
#
# Prerequisite: an NVIDIA GPU with the CUDA 12.1 toolkit (nvcc) installed — the kernel
# build needs nvcc on PATH.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ENV_NAME="${ENV_NAME:-pros_workflow}"
PYVER="${PYVER:-3.11}"
CUDA_TAG="${CUDA_TAG:-cu121}"
TORCH="${TORCH:-2.5.1}"
TVISION="${TVISION:-0.20.1}"

# --- make conda usable. `bash setup_nv.sh` is a non-interactive shell that often lacks
#     conda on PATH, and a broken ~/.local/bin/conda shim can shadow a real one, so test
#     that `conda info --base` actually works and otherwise source conda.sh from a common
#     install dir (all vars guarded with :- for `set -u`). ---
if ! conda info --base >/dev/null 2>&1; then
  for _s in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
            "/opt/conda" "${CONDA_EXE:-/nonexistent}/../.."; do
    [ -f "$_s/etc/profile.d/conda.sh" ] && { . "$_s/etc/profile.d/conda.sh"; break; }
  done
fi
conda info --base >/dev/null 2>&1 || {
  echo "ERROR: no working conda found. Run 'conda activate base' in a shell where conda" >&2
  echo "       works and re-run, or create the env manually (see README)." >&2; exit 1; }
. "$(conda info --base)/etc/profile.d/conda.sh"

# --- sanity: NVIDIA GPU + nvcc present ---
command -v nvidia-smi >/dev/null || echo "WARN: nvidia-smi not found — is the NVIDIA driver installed?" >&2
command -v nvcc >/dev/null || echo "WARN: nvcc not found — the pointnet2 kernel build needs the CUDA toolkit." >&2
echo ">> env: $ENV_NAME | python $PYVER | torch $TORCH+$CUDA_TAG"

# --- fail early if the env already exists (don't silently clobber) ---
if conda env list | grep -qE "^\s*$ENV_NAME\s|/$ENV_NAME\$"; then
  echo "ERROR: conda env '$ENV_NAME' already exists." >&2
  echo "       Remove it (conda env remove -n $ENV_NAME) or pass ENV_NAME=<other>." >&2
  exit 1
fi

R="conda run -n $ENV_NAME"

echo ">> [1/4] creating env"
conda create -n "$ENV_NAME" python="$PYVER" -y

echo ">> [2/4] installing CUDA PyTorch"
$R pip install "torch==${TORCH}+${CUDA_TAG}" "torchvision==${TVISION}+${CUDA_TAG}" \
    "torchaudio==${TORCH}+${CUDA_TAG}" --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo ">> [3/4] installing perception + GraspGen deps, then building the pointnet2 kernel"
$R pip install -r requirements-nv.txt
$R pip install --no-build-isolation -e tool/graspgen_runtime/pointnet2_ops

echo ">> [4/4] verifying GPU + kernel"
$R python -c "import torch; from pointnet2_ops import pointnet2_utils as u; \
print('GPU       :', torch.cuda.get_device_name(0)); \
print('pointnet2 :', tuple(u.furthest_point_sample(torch.randn(1,1024,3,device='cuda'),128).shape), 'ok')"

cat <<EOF

Done. The env '$ENV_NAME' is ready.

Launch the services (set EXTERNAL_IP to an address the Commander can reach):
  conda activate $ENV_NAME
  EXTERNAL_IP=<gpu-host> python -m get_item_info_agent_no_sam3d   # :8006 required
  EXTERNAL_IP=<gpu-host> python -m grasp_agent                    # :8007 required
  EXTERNAL_IP=<gpu-host> python -m get_item_info_agent            # :8008 optional (SAM3D)
EOF
