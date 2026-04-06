#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_IMAGE="vlm-rl-env:latest"
FORCE_REBUILD=false

for arg in "$@"; do
    case "$arg" in
        -b|--rebuild)
            FORCE_REBUILD=true
            ;;
    esac
done

ARCH="$(uname -m)"
OS="$(uname -s)"

echo "Detected OS: $OS, Architecture: $ARCH"

if [ "$FORCE_REBUILD" = true ] || ! docker image inspect "$LOCAL_IMAGE" >/dev/null 2>&1; then
    echo "Building local image '$LOCAL_IMAGE'..."
    docker build -t "$LOCAL_IMAGE" "$SCRIPT_DIR"
fi

if ! docker network ls --format '{{.Name}}' | grep -q '^compose_cube_bridge_network$'; then
    echo "Creating Docker bridge network 'compose_cube_bridge_network'..."
    docker network create --driver bridge compose_cube_bridge_network
fi

DOCKER_ARGS=(
    run
    -it
    --rm
    --network
    compose_cube_bridge_network
    --env
    OLLAMA_URL=http://140.116.82.233:11434
    --env
    OLLAMA_BASE_URL=http://140.116.82.233:11434
    -v
    "$SCRIPT_DIR:/workspaces/VLM_RL"
    -v
    vlm_rl_build:/workspaces/build
    -v
    vlm_rl_install:/workspaces/install
    -v
    vlm_rl_log:/workspaces/log
    -w
    /workspaces/VLM_RL
)

if [ -f "$SCRIPT_DIR/.env" ]; then
    DOCKER_ARGS+=(--env-file "$SCRIPT_DIR/.env")
fi

GPU_FLAGS=()
if [ "$OS" = "Linux" ]; then
    if [ -f "/etc/nv_tegra_release" ]; then
        GPU_FLAGS=(--runtime=nvidia)
    elif docker info --format '{{json .}}' 2>/dev/null | grep -q '"Runtimes".*nvidia'; then
        GPU_FLAGS=(--gpus all)
    fi
fi

read -r -d '' DOCKER_CMD <<'DOCKER_EOF' || true
set -e

export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_INSTALL_DIR=/workspaces/VLM_RL/.uv_python
export UV_PROJECT_ENVIRONMENT=/workspaces/VLM_RL/.venv_linux

if ! command -v uv >/dev/null 2>&1 || [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
    echo "Initializing uv environment..."
    pip install -q uv --no-warn-script-location 2>/dev/null
    uv python install 3.12 --quiet
    cd /workspaces/VLM_RL
    uv sync --frozen --no-dev --python 3.12 --quiet
fi

cat >/tmp/vlm_rl_shell_rc <<'RCFILE'
source /opt/ros/humble/setup.bash
source /workspaces/VLM_RL/container_env.sh
cd /workspaces/VLM_RL
RCFILE

echo "Container ready."
echo "Run 'ros_ws_build' once for ROS packages, then 'ros_ws_source'."
echo "Examples:"
echo "  ros2 run car_control_pkg car_control_node"
echo "  ros2 run arm_control_pkg arm_control_node"
echo "  ros2 launch vlm_rl_nav navigation.launch.py"

exec bash --noprofile --rcfile /tmp/vlm_rl_shell_rc -i
DOCKER_EOF

if [ "$ARCH" = "x86_64" ] || { [ "$ARCH" = "arm64" ] && [ "$OS" = "Darwin" ]; }; then
    docker "${DOCKER_ARGS[@]}" "${GPU_FLAGS[@]}" "$LOCAL_IMAGE" /bin/bash -lc "$DOCKER_CMD"
else
    docker "${DOCKER_ARGS[@]}" --runtime=nvidia "$LOCAL_IMAGE" /bin/bash -lc "$DOCKER_CMD"
fi
