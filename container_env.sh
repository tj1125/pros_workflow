export VLM_RL_ROOT=/workspaces/VLM_RL
export ROS_WS_ROOT=/workspaces
export ROS_WS_BUILD=/workspaces/build
export ROS_WS_INSTALL=/workspaces/install
export ROS_WS_LOG=/workspaces/log
export UV_PYTHON_INSTALL_DIR=/workspaces/VLM_RL/.uv_python
export UV_PROJECT_ENVIRONMENT=/workspaces/VLM_RL/.venv_linux
export PATH="$HOME/.local/bin:$PATH"

mock() {
    cd "$VLM_RL_ROOT" || return 1
    uv run python main.py --mock "$@"
}

run() {
    cd "$VLM_RL_ROOT" || return 1
    uv run python main.py --no-mock "$@"
}

t() {
    cd "$VLM_RL_ROOT" || return 1
    uv run python test_client.py --mock-only "$@"
}

logs() {
    python3 -m json.tool "$VLM_RL_ROOT/logs/trace_logger.jsonl" 2>/dev/null || echo "no logs yet"
}

ros_ws_build() {
    source /opt/ros/humble/setup.bash
    cd "$ROS_WS_ROOT" || return 1
    mkdir -p "$ROS_WS_BUILD" "$ROS_WS_INSTALL" "$ROS_WS_LOG"
    find "$ROS_WS_BUILD" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    find "$ROS_WS_INSTALL" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    find "$ROS_WS_LOG" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    colcon --log-base "$ROS_WS_LOG" build \
        --base-paths "$VLM_RL_ROOT/tools/nav" "$VLM_RL_ROOT/tools/car_control" \
        --build-base "$ROS_WS_BUILD" \
        --install-base "$ROS_WS_INSTALL" \
        --symlink-install \
        "$@"
    source "$ROS_WS_INSTALL/setup.bash"
}

r() {
    ros_ws_build "$@"
}

ros_ws_source() {
    source /opt/ros/humble/setup.bash
    if [ ! -f "$ROS_WS_INSTALL/setup.bash" ]; then
        echo "ROS overlay not built yet. Run: ros_ws_build"
        return 1
    fi
    if ! find "$ROS_WS_BUILD" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | grep -q .; then
        echo "ROS build space is empty. Run: ros_ws_build"
        return 1
    fi
    source "$ROS_WS_INSTALL/setup.bash"
}
