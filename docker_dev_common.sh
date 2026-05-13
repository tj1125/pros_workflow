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
WEB_PORT_SPEC=""

parse_common_docker_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            -b|--rebuild)
                FORCE_REBUILD=true
                shift
                ;;
            --web-port)
                if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                    echo "--web-port requires a port, for example: --web-port 8080" >&2
                    exit 1
                fi
                WEB_PORT_SPEC="$2"
                shift 2
                ;;
            --web-port=*)
                WEB_PORT_SPEC="${1#*=}"
                if [ -z "$WEB_PORT_SPEC" ]; then
                    echo "--web-port requires a port, for example: --web-port=8080" >&2
                    exit 1
                fi
                shift
                ;;
            *)
                echo "Unknown argument: $1" >&2
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

    if [ -n "$WEB_PORT_SPEC" ]; then
        local web_host_port="$WEB_PORT_SPEC"
        local web_container_port="$WEB_PORT_SPEC"

        if [[ "$WEB_PORT_SPEC" == *:* ]]; then
            web_host_port="${WEB_PORT_SPEC%%:*}"
            web_container_port="${WEB_PORT_SPEC##*:}"
        fi

        if [ -z "$web_host_port" ] || [ -z "$web_container_port" ]; then
            echo "Invalid --web-port value: $WEB_PORT_SPEC" >&2
            exit 1
        fi

        DOCKER_ARGS+=(
            -p
            "${web_host_port}:${web_container_port}"
            --env
            WEB_PORT="$web_container_port"
        )
    fi

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
