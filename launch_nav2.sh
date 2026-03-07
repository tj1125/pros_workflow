#!/bin/bash
# =============================================================================
# VLM-RL Nav2 Launcher
# =============================================================================
# 啟動 Nav2 導航系統（需要在 run 之前先啟動此指令）
#
# 使用方式：
#   ./launch_nav2.sh           -> 前景模式（建議除錯時使用，可直接看 Nav2 log）
#   ./launch_nav2.sh -d        -> 背景模式
#   ./launch_nav2.sh --stop    -> 停止 Nav2
# =============================================================================

set -e

COMPOSE_FILE="$(dirname "$0")/docker-compose-nav2.yml"
CONTAINER_NAME="vlm_rl_nav2"

# --- Parse arguments ---
DETACHED=""
STOP_MODE=false

for arg in "$@"; do
  case $arg in
    -d|--detach) DETACHED="-d" ;;
    --stop)      STOP_MODE=true ;;
  esac
done

# --- Ensure cube_bridge_network exists (shared by all ROS 2 containers) ---
for NETWORK in cube_bridge_network; do
  if ! docker network ls --format '{{.Name}}' | grep -q "^${NETWORK}$"; then
    echo "🌐 Creating Docker bridge network '${NETWORK}'..."
    docker network create --driver bridge "${NETWORK}"
  fi
done

# --- Stop mode ---
if [ "$STOP_MODE" = true ]; then
  echo "🛑 Stopping Nav2..."
  docker compose -f "$COMPOSE_FILE" down
  exit 0
fi

# --- Start Nav2 ---
echo "╔═══════════════════════════════════════════════════╗"
echo "║  VLM-RL Nav2  |  Starting Navigation Stack...     ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
echo "📄 Compose file: $COMPOSE_FILE"
echo "🐳 Container:    $CONTAINER_NAME"
echo ""

if [ -n "$DETACHED" ]; then
  docker compose -f "$COMPOSE_FILE" up -d
  echo "✅ Nav2 started in background."
  echo "   Logs: docker logs -f $CONTAINER_NAME"
  echo "   Stop: $0 --stop"
else
  echo "📡 Nav2 log output (press Ctrl+C to stop):"
  echo "────────────────────────────────────────────"
  docker compose -f "$COMPOSE_FILE" up
fi
