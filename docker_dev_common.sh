#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_IMAGE="vlm-rl-env:latest"
DEV_CONTAINER_LABEL="vlm_rl.dev_shell=1"

FORCE_REBUILD=false
ARCH="$(uname -m)"
OS="$(uname -s)"
GPU_FLAGS=()
DOCKER_ARGS=()

parse_common_docker_args() {
    for arg in "$@"; do
        case "$arg" in
            -b|--rebuild)
                FORCE_REBUILD=true
                ;;
            *)
                echo "Unknown argument: $arg" >&2
                exit 1
                ;;
        esac
    done
}

print_docker_environment() {
    echo "Detected OS: $OS, Architecture: $ARCH"
}

ensure_local_image() {
    if [ "$FORCE_REBUILD" = true ] || ! docker image inspect "$LOCAL_IMAGE" >/dev/null 2>&1; then
        echo "Building local image '$LOCAL_IMAGE'..."
        docker build -t "$LOCAL_IMAGE" "$SCRIPT_DIR"
    fi
}

ensure_bridge_network() {
    if ! docker network ls --format '{{.Name}}' | grep -q '^compose_cube_bridge_network$'; then
        echo "Creating Docker bridge network 'compose_cube_bridge_network'..."
        docker network create --driver bridge compose_cube_bridge_network
    fi
}

compute_gpu_flags() {
    GPU_FLAGS=()
    if [ "$OS" = "Linux" ]; then
        if [ -f "/etc/nv_tegra_release" ]; then
            GPU_FLAGS=(--runtime=nvidia)
        elif docker info --format '{{json .}}' 2>/dev/null | grep -q '"Runtimes".*nvidia'; then
            GPU_FLAGS=(--gpus all)
        fi
    fi
}

build_base_docker_args() {
    DOCKER_ARGS=(
        run
        -it
        --rm
        --label
        "$DEV_CONTAINER_LABEL"
        --network
        compose_cube_bridge_network
        --env
        OLLAMA_URL=http://192.168.75.24:11434
        --env
        OLLAMA_BASE_URL=http://192.168.75.24:11434
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
}

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

run_dev_container() {
    local extra_docker_args=("$@")
    local run_status=0

    set +e
    if [ "$ARCH" = "x86_64" ] || { [ "$ARCH" = "arm64" ] && [ "$OS" = "Darwin" ]; }; then
        docker "${DOCKER_ARGS[@]}" "${extra_docker_args[@]}" "${GPU_FLAGS[@]}" \
            "$LOCAL_IMAGE" --noprofile --rcfile /workspaces/VLM_RL/container_env.sh -i
        run_status=$?
    else
        docker "${DOCKER_ARGS[@]}" "${extra_docker_args[@]}" --runtime=nvidia \
            "$LOCAL_IMAGE" --noprofile --rcfile /workspaces/VLM_RL/container_env.sh -i
        run_status=$?
    fi
    set -e

    cleanup_dev_workspace
    return "$run_status"
}
