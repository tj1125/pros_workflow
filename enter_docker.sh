#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_IMAGE="vlm-rl-env:latest"
DEV_CONTAINER_LABEL="vlm_rl.dev_shell=1"
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

GPU_FLAGS=()
if [ "$OS" = "Linux" ]; then
    if [ -f "/etc/nv_tegra_release" ]; then
        GPU_FLAGS=(--runtime=nvidia)
    elif docker info --format '{{json .}}' 2>/dev/null | grep -q '"Runtimes".*nvidia'; then
        GPU_FLAGS=(--gpus all)
    fi
fi

DOCKER_ARGS=(
    run
    -it
    --rm
    --label
    "$DEV_CONTAINER_LABEL"
    --network
    compose_cube_bridge_network
    --env
    OLLAMA_URL=http://140.116.82.233:11434
    --env
    OLLAMA_BASE_URL=http://140.116.82.233:11434
    -v
    "$SCRIPT_DIR:/workspaces/VLM_RL"
    -v
    vlm_rl_dev_build:/workspaces/build
    -v
    vlm_rl_dev_install:/workspaces/install
    -v
    vlm_rl_dev_log:/workspaces/log
    -w
    /workspaces/VLM_RL
)

if [ -f "$SCRIPT_DIR/.env" ]; then
    DOCKER_ARGS+=(--env-file "$SCRIPT_DIR/.env")
fi

cleanup_dev_workspace() {
    if docker ps --filter "label=$DEV_CONTAINER_LABEL" --format '{{.ID}}' | grep -q .; then
        return 0
    fi

    echo "Clearing dev ROS workspace volumes..."
    docker run --rm \
        -v vlm_rl_dev_build:/workspaces/build \
        -v vlm_rl_dev_install:/workspaces/install \
        -v vlm_rl_dev_log:/workspaces/log \
        "$LOCAL_IMAGE" \
        -lc '
            mkdir -p /workspaces/build /workspaces/install /workspaces/log
            find /workspaces/build -mindepth 1 -maxdepth 1 -exec rm -rf {} +
            find /workspaces/install -mindepth 1 -maxdepth 1 -exec rm -rf {} +
            find /workspaces/log -mindepth 1 -maxdepth 1 -exec rm -rf {} +
        '
}

set +e
if [ "$ARCH" = "x86_64" ] || { [ "$ARCH" = "arm64" ] && [ "$OS" = "Darwin" ]; }; then
    docker "${DOCKER_ARGS[@]}" "${GPU_FLAGS[@]}" "$LOCAL_IMAGE" --noprofile --rcfile /workspaces/VLM_RL/container_env.sh -i
else
    docker "${DOCKER_ARGS[@]}" --runtime=nvidia "$LOCAL_IMAGE" --noprofile --rcfile /workspaces/VLM_RL/container_env.sh -i
fi
RUN_STATUS=$?
set -e

cleanup_dev_workspace
exit "$RUN_STATUS"
