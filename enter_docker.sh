#!/bin/bash
# VLM-RL System — Enter ROS Container
#
# First run : automatically installs uv + Python dependencies into .venv
# Subsequent runs : .venv already exists, skips setup
#
# Usage: ./enter_docker.sh

SETUP_CMD='
# --- VLM-RL auto-setup (runs only if .venv is missing) ---
if [ ! -d /workspaces/VLM_RL/.venv ]; then
  echo "⚙️  First run: setting up Python environment..."
  pip install -q uv
  cd /workspaces/VLM_RL && uv sync --frozen --no-dev
  echo "✅ Setup complete!"
fi

# Aliases
alias r="colcon build --symlink-install"
alias run="cd /workspaces/VLM_RL && uv run python main.py"
alias mock="cd /workspaces/VLM_RL && uv run python main.py --mock"
alias t="cd /workspaces/VLM_RL && uv run python test_client.py --mock-only"
alias logs="cat /workspaces/VLM_RL/logs/trace_logger.jsonl | python3 -m json.tool 2>/dev/null || echo (no logs yet)"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  VLM-RL  |  ROS Container  |  /workspaces/VLM_RL  ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  mock  → run main.py in mock mode                   ║"
echo "║  run   → run main.py in real mode (needs .env)      ║"
echo "║  t     → run test_client.py (mock-only)             ║"
echo "║  r     → colcon build --symlink-install              ║"
echo "║  logs  → print trace log                            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

cd /workspaces/VLM_RL
exec bash
'

docker run -it --rm \
  -v "$(pwd):/workspaces/VLM_RL" \
  -w /workspaces/VLM_RL \
  --network compose_cube_bridge_network \
  --env-file ./.env \
  --env OLLAMA_URL="http://140.116.82.233:11434" \
  --env OLLAMA_BASE_URL="http://140.116.82.233:11434" \
  registry.screamtrumpet.csie.ncku.edu.tw/unity_env/pros_rl_image:latest \
  /bin/bash -c "$SETUP_CMD"
