#!/bin/bash
# VLM-RL System — Enter ROS Container
#
# Uses pros_rl_image (ROS 2 environment)
# Python dependencies are installed into .venv_linux (Linux-compatible, separate from macOS .venv)
#
# Usage: ./enter_docker.sh

# Inner bash script executed inside the container
read -r -d '' INNER << 'EOF'
echo "📦 Installing uv..."
pip install -q uv

echo "📦 Downloading Python 3.12 (if missing)..."
uv python install 3.12

echo "📦 Syncing Python dependencies (Linux venv)..."
export UV_PROJECT_ENVIRONMENT=/workspaces/VLM_RL/.venv_linux
cd /workspaces/VLM_RL
uv sync --frozen --no-dev --python 3.12

# Write persistent settings to bashrc
cat >> ~/.bashrc << 'BASHRC'
export UV_PROJECT_ENVIRONMENT=/workspaces/VLM_RL/.venv_linux
export PATH="$HOME/.local/bin:$PATH"
alias mock="cd /workspaces/VLM_RL && uv run python main.py --mock"
alias run="cd /workspaces/VLM_RL && uv run python main.py --no-mock"
alias t="cd /workspaces/VLM_RL && uv run python test_client.py --mock-only"
alias r="colcon build --symlink-install"
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

exec bash
EOF

docker run -it --rm \
  -v "$(pwd):/workspaces/VLM_RL" \
  -w /workspaces/VLM_RL \
  --network compose_cube_bridge_network \
  --env-file ./.env \
  --env OLLAMA_URL="http://140.116.82.233:11434" \
  --env OLLAMA_BASE_URL="http://140.116.82.233:11434" \
  registry.screamtrumpet.csie.ncku.edu.tw/unity_env/pros_rl_image:latest \
  /bin/bash -c "$INNER"
