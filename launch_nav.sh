#!/bin/bash

set -euo pipefail

COMPOSE_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/docker-compose-nav2.yml"
CONTAINER_NAME="vlm_rl_nav2"
NAV_IMAGE="vlm-rl-nav2:latest"
NAV_SERVICE="nav2"

DETACHED=""
STOP_MODE=false
WITH_ROSBRIDGE=false
REBUILD_IMAGE=false

for arg in "$@"; do
    case "$arg" in
        -d|--detach)
            DETACHED="-d"
            ;;
        --stop)
            STOP_MODE=true
            ;;
        --rosbridge)
            WITH_ROSBRIDGE=true
            ;;
        --rebuild)
            REBUILD_IMAGE=true
            ;;
    esac
done

for network in compose_cube_bridge_network cube_bridge_network; do
    if ! docker network ls --format '{{.Name}}' | grep -q "^${network}$"; then
        echo "Creating Docker bridge network '${network}'..."
        docker network create --driver bridge "${network}"
    fi
done

if [ "$STOP_MODE" = true ]; then
    echo "Stopping Nav container..."
    docker compose -f "$COMPOSE_FILE" down
    exit 0
fi

if [ "$REBUILD_IMAGE" = true ] || ! docker image inspect "$NAV_IMAGE" >/dev/null 2>&1; then
    echo "Building local Nav image '$NAV_IMAGE'..."
    docker compose -f "$COMPOSE_FILE" build "$NAV_SERVICE"
fi

COMPOSE_PREFIX=()
if [ "$WITH_ROSBRIDGE" = true ]; then
    COMPOSE_PREFIX=(env COMPOSE_PROFILES=with-rosbridge)
    echo "Rosbridge profile enabled."
fi

echo "Starting Nav stack from $COMPOSE_FILE"

if [ -n "$DETACHED" ]; then
    "${COMPOSE_PREFIX[@]}" docker compose -f "$COMPOSE_FILE" up -d
    echo "Nav stack started in background."
    echo "Logs: docker logs -f $CONTAINER_NAME"
    echo "Stop: $0 --stop"
else
    "${COMPOSE_PREFIX[@]}" docker compose -f "$COMPOSE_FILE" up
fi
