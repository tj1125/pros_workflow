#!/usr/bin/env bash
# setup_rocm.sh — one self-contained command to build the AMD ROCm conda env for the
# :8006/:8007 perception + grasp services (the AMD counterpart of the CUDA setup).
#
# Usage (from this folder):
#   bash setup_rocm.sh
# Optional overrides:
#   ENV_NAME=pros_workflow PYVER=3.12 GFX=gfx1150 bash setup_rocm.sh
#
# It: creates the conda env, installs the ROCm PyTorch + SDK from AMD's per-GPU wheel
# index, installs the perception/GraspGen deps (requirements-rocm.txt), installs a hipcc
# wrapper into the SDK (embedded below — no external script needed), builds the GraspGen
# pointnet2 kernel, and verifies the GPU works. The legacy SAM3D service (:8008) is not
# covered (not ported to ROCm).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ENV_NAME="${ENV_NAME:-pros_workflow}"
PYVER="${PYVER:-3.12}"

# --- locate conda ---
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
[ -n "$CONDA_BASE" ] && source "$CONDA_BASE/etc/profile.d/conda.sh" || {
  echo "ERROR: conda not found on PATH. Install Miniconda first." >&2; exit 1; }

# --- detect the GPU arch (gfx1151 = Radeon 8060S, gfx1150 = 860M/890M, ...) ---
if [ -z "${GFX:-}" ]; then
  GFX="$(rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-f]\+' || true)"
  GFX="${GFX:-gfx1151}"
fi
INDEX="https://repo.amd.com/rocm/whl/$GFX/"
echo ">> GPU arch: $GFX | env: $ENV_NAME | python $PYVER"
echo ">> wheel index: $INDEX"

# --- fail early if the env already exists (don't silently clobber) ---
if conda env list | grep -qE "^\s*$ENV_NAME\s|/$ENV_NAME\$"; then
  echo "ERROR: conda env '$ENV_NAME' already exists." >&2
  echo "       Remove it (conda env remove -n $ENV_NAME) or pass ENV_NAME=<other>." >&2
  exit 1
fi

R="conda run -n $ENV_NAME"

echo ">> [1/5] creating env"
conda create -n "$ENV_NAME" python="$PYVER" -y

echo ">> [2/5] installing ROCm PyTorch + SDK from AMD index"
$R pip install --index-url "$INDEX" --extra-index-url https://pypi.org/simple \
    torch torchvision torchaudio
# match rocm[devel] to the rocm-sdk-core version torch pulled in, then expand it
ROCM_VER="$($R python -c 'import importlib.metadata as m; print(m.version("rocm-sdk-core"))')"
echo "   rocm-sdk-core == $ROCM_VER"
$R pip install --index-url "$INDEX" --extra-index-url https://pypi.org/simple "rocm[devel]==$ROCM_VER"
$R rocm-sdk init

echo ">> [3/5] installing perception + GraspGen deps"
$R pip install -r requirements-rocm.txt

echo ">> [4/5] installing hipcc wrapper + building the pointnet2 kernel for $GFX"
DEVEL="$($R python -c 'import _rocm_sdk_devel,os;print(os.path.dirname(_rocm_sdk_devel.__file__))')"
BIN="$DEVEL/bin"

# Install the hipcc wrapper. Building PyTorch HIP extensions against the pip-packaged
# ROCm SDK needs hipcc's compile invocation normalized (single `-x hip` before the
# source, SDK include first, ROCm env vars cleared) so the SDK's HIP headers are used
# instead of a stale system copy under /usr/include/hip — otherwise the build fails on
# duplicate abort/__assert_fail and unresolved hipsolver/fp8 types. We swap the SDK's
# real hipcc for a wrapper (keeping the original as hipcc.real). Idempotent.
if [ ! -e "$BIN/hipcc.real" ]; then
  cp -a "$BIN/hipcc" "$BIN/hipcc.real"
  echo "   backed up real hipcc -> hipcc.real"
fi
cat > "$BIN/hipcc" <<'HIPCC_WRAPPER'
#!/usr/bin/env bash
# hipcc wrapper (installed by setup_rocm.sh) — normalizes PyTorch's HIP-extension
# compile invocation so the SDK's HIP headers are used consistently instead of the
# stale system copy under /usr/include/hip. Only .hip/.cu device sources are forced to
# HIP; anything else is forwarded verbatim to the real compiler (hipcc.real).
set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL="$SELF_DIR/hipcc.real"
SDK_INC="$(cd "$SELF_DIR/../include" && pwd)"

# hipcc.real mis-resolves headers when ROCM_PATH/HIP_PATH point at the SDK; letting it
# auto-detect its own location (inside the SDK) is what works.
unset ROCM_HOME ROCM_PATH HIP_PATH HIP_ROOT_DIR HIP_CLANG_PATH 2>/dev/null || true

includes=(); declare -A seen_inc=(); rest=(); src=""; out=""
args=("$@"); n=${#args[@]}; i=0
while [ $i -lt $n ]; do
  a="${args[$i]}"
  case "$a" in
    -I?*) dir="${a#-I}"; if [ -z "${seen_inc[$dir]+x}" ]; then seen_inc[$dir]=1; includes+=("-I$dir"); fi ;;
    -I)   dir="${args[$((i+1))]}"; i=$((i+1)); if [ -z "${seen_inc[$dir]+x}" ]; then seen_inc[$dir]=1; includes+=("-I$dir"); fi ;;
    -o)   out="${args[$((i+1))]}"; i=$((i+1)) ;;
    -o?*) out="${a#-o}" ;;
    -c)   : ;;
    *.hip|*.cu) src="$a" ;;
    *)    rest+=("$a") ;;
  esac
  i=$((i+1))
done

if [ -z "$src" ]; then exec "$REAL" "$@"; fi

# SDK include first + exactly once; `-c -x hip <src>` then output, then the rest of the
# flags trailing after the source (this exact shape is what hipcc.real handles).
final_inc=("-I$SDK_INC")
for x in "${includes[@]}"; do [ "$x" = "-I$SDK_INC" ] || final_inc+=("$x"); done
cmd=("$REAL" "${final_inc[@]}" -c -x hip "$src")
[ -n "$out" ] && cmd+=(-o "$out")
cmd+=("${rest[@]}")
exec "${cmd[@]}"
HIPCC_WRAPPER
chmod +x "$BIN/hipcc"
echo "   installed hipcc wrapper -> $BIN/hipcc"

# build the kernel (torch's hipify still needs ROCM_HOME set; the wrapper clears it
# again before calling hipcc.real)
$R env ROCM_HOME="$DEVEL" ROCM_PATH="$DEVEL" HIP_PATH="$DEVEL" PYTORCH_ROCM_ARCH="$GFX" \
    pip install --no-build-isolation -e tool/graspgen_runtime/pointnet2_ops

echo ">> [5/5] verifying GPU + kernel"
$R python -c "import torch; from pointnet2_ops import pointnet2_utils as u; \
print('GPU       :', torch.cuda.get_device_name(0)); \
print('pointnet2 :', tuple(u.furthest_point_sample(torch.randn(1,1024,3,device='cuda'),128).shape), 'ok')"

cat <<EOF

Done. The env '$ENV_NAME' is ready.

Launch the two required services (set EXTERNAL_IP to an address the Commander can reach):
  conda activate $ENV_NAME
  EXTERNAL_IP=<gpu-host> python -m get_item_info_agent_no_sam3d   # :8006
  EXTERNAL_IP=<gpu-host> python -m grasp_agent                    # :8007
EOF
