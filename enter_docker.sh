#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/docker_dev_common.sh"

parse_common_docker_args "$@"
print_docker_environment
ensure_local_image
ensure_bridge_network
compute_gpu_flags
build_base_docker_args
run_dev_container
exit $?
