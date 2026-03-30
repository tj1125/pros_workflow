#!/bin/bash
# VLM-RL System — Enter Dev Container

LOCAL_IMAGE="vlm-rl-env:latest"
VOLUME_ARGS="-v $(pwd):/workspaces/VLM_RL -v $(pwd)/tools/nav:/workspaces/tools/nav -v $(pwd)/tools/pros_car:/workspaces/tools/pros_car -v vlm_rl_nav_build:/workspaces/nav_build -v vlm_rl_nav_install:/workspaces/nav_install -v vlm_rl_nav_log:/workspaces/nav_log"

# --- Detect OS and Architecture ---
ARCH=$(uname -m)
OS=$(uname -s)
echo "Detected OS: $OS, Architecture: $ARCH"

# --- Auto-build local image if not present (only needs to run once ever) ---
if ! docker image inspect "$LOCAL_IMAGE" > /dev/null 2>&1; then
    echo "🔨 Building local image '$LOCAL_IMAGE' (first time only, ~30s)..."
    docker build -t "$LOCAL_IMAGE" "$(dirname "$0")"
fi

# --- Common Docker Arguments ---
COMMON_ARGS="--network compose_cube_bridge_network --env OLLAMA_URL=http://140.116.82.233:11434 --env OLLAMA_BASE_URL=http://140.116.82.233:11434 $VOLUME_ARGS -w /workspaces/VLM_RL"
if [ -f "./.env" ]; then
    COMMON_ARGS+=" --env-file ./.env"
fi

# --- Detect GPU (Linux only) ---
GPU_FLAGS=""
if [ "$OS" = "Linux" ]; then
    if [ -f "/etc/nv_tegra_release" ]; then
        GPU_FLAGS="--runtime=nvidia"
    elif docker info --format '{{json .}}' 2>/dev/null | grep -q '"Runtimes".*nvidia'; then
        GPU_FLAGS="--gpus all"
    fi
fi

# --- Script executed inside the container ---
read -r -d '' DOCKER_CMD << 'DOCKER_EOF'
export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_INSTALL_DIR=/workspaces/VLM_RL/.uv_python
export UV_PROJECT_ENVIRONMENT=/workspaces/VLM_RL/.venv_linux

# 1. Python / uv setup (first-run only, results persist via volume mount)
if ! command -v uv &> /dev/null || [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
    echo "📦 Initializing environment (first run only)..."
    pip install -q uv --no-warn-script-location 2>/dev/null
    uv python install 3.12 --quiet
    cd /workspaces/VLM_RL && uv sync --frozen --no-dev --python 3.12 --quiet
fi

# 2. Aliases (written to volume to survive across sessions)
cat > /workspaces/VLM_RL/.container_env.sh << 'ENVFILE'
export UV_PYTHON_INSTALL_DIR=/workspaces/VLM_RL/.uv_python
export UV_PROJECT_ENVIRONMENT=/workspaces/VLM_RL/.venv_linux
export PATH="$HOME/.local/bin:$PATH"
alias mock="cd /workspaces/VLM_RL && uv run python main.py --mock"
alias run="cd /workspaces/VLM_RL && uv run python main.py --no-mock"
alias t="cd /workspaces/VLM_RL && uv run python test_client.py --mock-only"
alias r="cd /workspaces && find /workspaces/nav_build -mindepth 1 -maxdepth 1 -exec rm -rf {} + && find /workspaces/nav_install -mindepth 1 -maxdepth 1 -exec rm -rf {} + && find /workspaces/nav_log -mindepth 1 -maxdepth 1 -exec rm -rf {} + && colcon --log-base /workspaces/nav_log build --base-paths /workspaces/tools/nav /workspaces/tools/pros_car --build-base /workspaces/nav_build --install-base /workspaces/nav_install --packages-select action_interface car_control_pkg nav_goal_bridge_pkg vlm_rl_nav --symlink-install && source /workspaces/nav_install/setup.bash && cd /workspaces/VLM_RL"
alias logs='cat /workspaces/VLM_RL/logs/trace_logger.jsonl | python3 -m json.tool 2>/dev/null || echo "no logs yet"'
ENVFILE

grep -q ".container_env.sh" ~/.bashrc || echo "source /workspaces/VLM_RL/.container_env.sh" >> ~/.bashrc
source /workspaces/VLM_RL/.container_env.sh

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  VLM-RL  |  ROS Container  |  /workspaces/VLM_RL ║"
echo "╠═══════════════════════════════════════════════════╣"
echo "║  mock   → main.py --mock                         ║"
echo "║  run    → main.py --no-mock (needs real .env)    ║"
echo "║  t      → test_client.py --mock-only             ║"
echo "║  r      → build nav tool workspace               ║"
echo "║  logs   → print trace log                        ║"
echo "╚═══════════════════════════════════════════════════╝"

cd /workspaces/VLM_RL
exec bash
DOCKER_EOF

# --- Run Docker ---
if [ "$ARCH" = "x86_64" ] || ([ "$ARCH" = "arm64" ] && [ "$OS" = "Darwin" ]); then
    if [ -n "$GPU_FLAGS" ]; then
        docker run -it --rm $COMMON_ARGS $GPU_FLAGS "$LOCAL_IMAGE" /bin/bash -c "$DOCKER_CMD"
    else
        docker run -it --rm $COMMON_ARGS "$LOCAL_IMAGE" /bin/bash -c "$DOCKER_CMD"
    fi
else
    # arm64 Linux (e.g. Jetson)
    docker run -it --rm $COMMON_ARGS --runtime=nvidia "$LOCAL_IMAGE" /bin/bash -c "$DOCKER_CMD"
fi
