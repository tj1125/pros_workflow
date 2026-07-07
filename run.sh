#!/bin/bash

DOCKER_IMAGE="pros_workflow_image:latest"
DOCKER_CONTAINER="pros_workflow"
DOCKER_NETWORK="compose_cube_bridge_network"

cd "$(dirname "$0")" || exit 1

if [ ! -f .env ]; then
    echo "Missing .env. Create it with the workflow settings first."
    exit 1
fi

# Create a network if it does not exist.
if [ -z "$(docker network ls --filter name=$DOCKER_NETWORK --quiet)" ]; then
    docker network create $DOCKER_NETWORK
fi

# Re-enter the running workflow container from another terminal.
if [ -n "$(docker ps --filter name=$DOCKER_CONTAINER --quiet)" ]; then
    docker exec -it -w /workspace/pros_workflow $DOCKER_CONTAINER /bin/bash
    exit 0
fi

docker run -it --rm \
        --privileged \
        --name $DOCKER_CONTAINER \
        --network $DOCKER_NETWORK \
        -p 8080:8080 \
        -p 9090:9090 \
        -v $(pwd)/src/:/workspaces/src \
        -v $(pwd)/:/workspace/pros_workflow \
        -v pros_workflow_build:/workspaces/build \
        -v pros_workflow_install:/workspaces/install \
        -v pros_workflow_log:/workspaces/log \
        --shm-size=2048m \
        --env-file $(pwd)/.env \
        -w /workspace/pros_workflow \
        $DOCKER_IMAGE \
        /bin/bash
