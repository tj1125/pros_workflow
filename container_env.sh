export VLM_RL_ROOT=/workspaces/VLM_RL
export ROS_WS_ROOT=/workspaces
export ROS_WS_BUILD=/workspaces/build
export ROS_WS_INSTALL=/workspaces/install
export ROS_WS_LOG=/workspaces/log
export UV_PYTHON_INSTALL_DIR=/workspaces/VLM_RL/.uv_python
export UV_PROJECT_ENVIRONMENT=/workspaces/VLM_RL/.venv_linux
export PATH="$HOME/.local/bin:$PATH"

_ros_source_base() {
    source /opt/ros/humble/setup.bash
}

_ros_overlay_ready() {
    [ -f "$ROS_WS_INSTALL/setup.bash" ] &&
        find "$ROS_WS_BUILD" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | grep -q .
}

_ros_clear_overlay_env() {
    unset AMENT_PREFIX_PATH
    unset CMAKE_PREFIX_PATH
    unset COLCON_PREFIX_PATH
}

mock() {
    cd "$VLM_RL_ROOT" || return 1
    uv run python main.py --mock "$@"
}

run() {
    cd "$VLM_RL_ROOT" || return 1
    if [ -n "${WEB_PORT:-}" ]; then
        uv run python web_main.py --host "${WEB_HOST:-0.0.0.0}" --port "$WEB_PORT" "$@"
        return $?
    fi
    uv run python main.py --no-mock "$@"
}

web() {
    cd "$VLM_RL_ROOT" || return 1
    uv run python web_main.py --host "${WEB_HOST:-0.0.0.0}" --port "${WEB_PORT:-8080}" "$@"
}

t() {
    cd "$VLM_RL_ROOT" || return 1
    uv run python test_client.py --mock-only "$@"
}

logs() {
    python3 -m json.tool "$VLM_RL_ROOT/logs/trace_logger.jsonl" 2>/dev/null || echo "no logs yet"
}

ros_ws_source() {
    _ros_source_base
    if ! _ros_overlay_ready; then
        echo "ROS workspace not built yet. Run: r"
        return 1
    fi
    source "$ROS_WS_INSTALL/setup.bash"
}

ros_ws_build() {
    (
        _ros_clear_overlay_env
        _ros_source_base
        cd "$ROS_WS_ROOT" || exit 1
        mkdir -p "$ROS_WS_BUILD" "$ROS_WS_INSTALL" "$ROS_WS_LOG"
        find "$ROS_WS_BUILD" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
        find "$ROS_WS_INSTALL" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
        find "$ROS_WS_LOG" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
        colcon --log-base "$ROS_WS_LOG" build \
            --base-paths "$VLM_RL_ROOT/tools/car_control" \
            --build-base "$ROS_WS_BUILD" \
            --install-base "$ROS_WS_INSTALL" \
            "$@"
    ) || return 1

    ros_ws_source
}

r() {
    ros_ws_build "$@"
}

if [[ $- == *i* ]]; then
    _ros_source_base
    ros_ws_source >/dev/null 2>&1 || true
    if [ -r /root/.oh-my-bash/oh-my-bash.sh ]; then
        export OSH=/root/.oh-my-bash
        OSH_THEME='agnoster'
        source "$OSH"/oh-my-bash.sh
    else
        PS1='\[\E[0m\]\[\E[0;40m\] \[\E[33m\]⚡\[\E[0;40m\] root@\h \[\E[30;44m\]\[\E[0;44;30m\] \w \[\E[0;34m\] \[\E[0m\]'
    fi
    cd "$VLM_RL_ROOT" || :
fi
