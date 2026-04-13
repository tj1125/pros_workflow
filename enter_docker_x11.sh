#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/docker_dev_common.sh"

BASE_LOCAL_IMAGE="$LOCAL_IMAGE"
X11_OMPL_IMAGE="vlm-rl-env:x11-ompl"
OMPL_PYTHON_VERSION="1.7.0"
X11_EXTRA_PYTHON_PACKAGES="Pillow PyYAML"

X11_ACCESS_GRANTED=false
X11_DOCKER_ARGS=()

cleanup_x11_access() {
    if [ "$X11_ACCESS_GRANTED" = true ] && command -v xhost >/dev/null 2>&1; then
        xhost -SI:localuser:root >/dev/null 2>&1 || true
    fi
}

require_linux_x11_host() {
    if [ "$OS" != "Linux" ]; then
        echo "enter_docker_x11.sh currently supports Linux hosts only." >&2
        exit 1
    fi

    if [ -z "${DISPLAY:-}" ]; then
        echo "DISPLAY is not set. Launch this script from a local desktop terminal with X11 available." >&2
        exit 1
    fi

    if [ ! -d /tmp/.X11-unix ]; then
        echo "/tmp/.X11-unix is missing on the host, so X11 socket forwarding cannot be enabled." >&2
        exit 1
    fi

    if ! command -v xhost >/dev/null 2>&1; then
        echo "xhost is required on the host to grant temporary X11 access." >&2
        exit 1
    fi
}

build_x11_docker_args() {
    X11_DOCKER_ARGS=(
        --env DISPLAY="$DISPLAY"
        --env QT_X11_NO_MITSHM=1
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    )
}

grant_x11_access() {
    xhost +SI:localuser:root >/dev/null
    X11_ACCESS_GRANTED=true
    echo "Granted temporary X11 access for local root containers."
}

ensure_x11_ompl_image() {
    ensure_local_image

    if [ "$FORCE_REBUILD" = true ] || ! docker image inspect "$X11_OMPL_IMAGE" >/dev/null 2>&1; then
        echo "Building X11 image '$X11_OMPL_IMAGE' with OMPL Python bindings..."
        docker build -t "$X11_OMPL_IMAGE" - <<EOF
FROM $BASE_LOCAL_IMAGE

RUN python3 -m pip install --no-cache-dir ompl==$OMPL_PYTHON_VERSION
RUN python3 -m pip install --no-cache-dir $X11_EXTRA_PYTHON_PACKAGES
EOF
    fi

    LOCAL_IMAGE="$X11_OMPL_IMAGE"
}

parse_common_docker_args "$@"
print_docker_environment
require_linux_x11_host
ensure_x11_ompl_image
ensure_bridge_network
compute_gpu_flags
build_base_docker_args
build_x11_docker_args
grant_x11_access
trap cleanup_x11_access EXIT
run_dev_container "${X11_DOCKER_ARGS[@]}"
exit $?
