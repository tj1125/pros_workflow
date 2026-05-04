#!/bin/bash
# http://host.docker.internal:11434  # for local ollama

VOLUME_ARGS="-v $(pwd)/AI_pkg:/workspaces/AI_pkg"

ARCH=$(uname -m)
OS=$(uname -s)

GPU_FLAGS=""
USE_GPU=false

if [ "$OS" = "Linux" ]; then
    if [ -f "/etc/nv_tegra_release" ]; then
        GPU_FLAGS="--runtime=nvidia"
        USE_GPU=true
    elif docker info --format '{{json .}}' | grep -q '"Runtimes".*nvidia'; then
        GPU_FLAGS="--gpus all"
        USE_GPU=true
    fi
fi

if [ "$USE_GPU" = true ]; then
    echo "Testing Docker run with GPU..."
    docker run --rm $GPU_FLAGS registry.screamtrumpet.csie.ncku.edu.tw/unity_env/pros_rl_image:latest /bin/bash -c "echo GPU test" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "GPU not supported or failed, disabling GPU flags."
        GPU_FLAGS=""
        USE_GPU=false
    fi
fi

echo "Detected OS: $OS, Architecture: $ARCH"
echo "GPU Flags: $GPU_FLAGS"

DOCKER_CMD="cd /workspaces/AI_pkg && bash"

if [ "$ARCH" = "aarch64" ]; then
    echo "Detected architecture: arm64"
    docker run -it --rm \
        --network compose_cube_bridge_network \
        --runtime=nvidia \
        --env-file ./.env \
        --env OLLAMA_URL="http://192.168.75.24:11434" \
        $VOLUME_ARGS \
        registry.screamtrumpet.csie.ncku.edu.tw/unity_env/pros_rl_image:latest \
        bash -c "$DOCKER_CMD"

elif [ "$ARCH" = "x86_64" ] || ([ "$ARCH" = "arm64" ] && [ "$OS" = "Darwin" ]); then
    echo "Detected architecture: amd64 or macOS arm64"

    echo "Trying to run with GPU support..."
    docker run -it --rm \
        --network compose_cube_bridge_network \
        $GPU_FLAGS \
        --env-file ./.env \
        --env OLLAMA_URL="http://192.168.75.24:11434" \
        $VOLUME_ARGS \
        registry.screamtrumpet.csie.ncku.edu.tw/unity_env/pros_rl_image:latest \
        bash -c "$DOCKER_CMD"

    if [ $? -ne 0 ]; then
        echo "GPU not supported or failed, falling back to CPU mode..."
        docker run -it --rm \
            --network compose_cube_bridge_network \
            --env-file ./.env \
            --env OLLAMA_URL="http://192.168.75.24:11434" \
            $VOLUME_ARGS \
            registry.screamtrumpet.csie.ncku.edu.tw/unity_env/pros_rl_image:latest \
            bash -c "$DOCKER_CMD"
    fi
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi
