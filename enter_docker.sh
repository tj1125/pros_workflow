#!/bin/bash
# VLM-RL System — Enter ROS Container
# 結合 GPU 偵測機制 (來自 AI_pkg 的 ai_docker_pc.sh)

VOLUME_ARGS="-v $(pwd):/workspaces/VLM_RL"

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

# Inner bash script executed inside the container
read -r -d '' DOCKER_CMD << 'EOF'
# Setup uv paths and persistent Python install directory
export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_INSTALL_DIR=/workspaces/VLM_RL/.uv_python
export UV_PROJECT_ENVIRONMENT=/workspaces/VLM_RL/.venv_linux

# Check if environment is already fully set up to skip redundant steps
if ! command -v uv &> /dev/null || [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
    echo "📦 Initializing VLM-RL Environment (First run might take a minute)..."

    echo "   📥 Installing uv..."
    pip install -q uv

    echo "   � Downloading Python 3.12..."
    uv python install 3.12

    echo "   � Syncing Python dependencies (Linux venv)..."
    cd /workspaces/VLM_RL
    uv sync --frozen --no-dev --python 3.12
fi

# Write persistent settings to bashrc
cat >> ~/.bashrc << 'BASHRC'
export UV_PYTHON_INSTALL_DIR=/workspaces/VLM_RL/.uv_python
export UV_PROJECT_ENVIRONMENT=/workspaces/VLM_RL/.venv_linux
export PATH="$HOME/.local/bin:$PATH"
alias mock="cd /workspaces/VLM_RL && uv run python main.py --mock"
alias run="cd /workspaces/VLM_RL && uv run python main.py --no-mock"
alias t="cd /workspaces/VLM_RL && uv run python test_client.py --mock-only"
alias r="cd /workspaces && colcon build --base-paths /workspaces/VLM_RL/vendor --symlink-install && source /workspaces/install/setup.bash && cd /workspaces/VLM_RL"
alias logs='cat /workspaces/VLM_RL/logs/trace_logger.jsonl | python3 -m json.tool 2>/dev/null || echo "no logs yet"'
BASHRC

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  VLM-RL  |  ROS Container  |  /workspaces/VLM_RL ║"
echo "╠═══════════════════════════════════════════════════╣"
echo "║  mock   → main.py --mock                         ║"
echo "║  run    → main.py --no-mock (needs real .env)    ║"
echo "║  t      → test_client.py --mock-only             ║"
echo "║  r      → colcon build --symlink-install         ║"
echo "║  logs   → print trace log                        ║"
echo "╚═══════════════════════════════════════════════════╝"

cd /workspaces/VLM_RL
exec bash
EOF

if [ "$ARCH" = "aarch64" ]; then
    echo "Detected architecture: arm64"
    docker run -it --rm \
        --network compose_cube_bridge_network \
        --runtime=nvidia \
        --env-file ./.env \
        --env OLLAMA_URL="http://140.116.82.233:11434" \
        --env OLLAMA_BASE_URL="http://140.116.82.233:11434" \
        $VOLUME_ARGS \
        -w /workspaces/VLM_RL \
        registry.screamtrumpet.csie.ncku.edu.tw/unity_env/pros_rl_image:latest \
        /bin/bash -c "$DOCKER_CMD"

elif [ "$ARCH" = "x86_64" ] || ([ "$ARCH" = "arm64" ] && [ "$OS" = "Darwin" ]); then
    echo "Detected architecture: amd64 or macOS arm64"

    echo "Trying to run with GPU support..."
    docker run -it --rm \
        --network compose_cube_bridge_network \
        $GPU_FLAGS \
        --env-file ./.env \
        --env OLLAMA_URL="http://140.116.82.233:11434" \
        --env OLLAMA_BASE_URL="http://140.116.82.233:11434" \
        $VOLUME_ARGS \
        -w /workspaces/VLM_RL \
        registry.screamtrumpet.csie.ncku.edu.tw/unity_env/pros_rl_image:latest \
        /bin/bash -c "$DOCKER_CMD"

    if [ $? -ne 0 ]; then
        echo "GPU not supported or failed, falling back to CPU mode..."
        docker run -it --rm \
            --network compose_cube_bridge_network \
            --env-file ./.env \
            --env OLLAMA_URL="http://140.116.82.233:11434" \
            --env OLLAMA_BASE_URL="http://140.116.82.233:11434" \
            $VOLUME_ARGS \
            -w /workspaces/VLM_RL \
            registry.screamtrumpet.csie.ncku.edu.tw/unity_env/pros_rl_image:latest \
            /bin/bash -c "$DOCKER_CMD"
    fi
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi
