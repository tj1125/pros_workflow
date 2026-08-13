export PROS_WORKFLOW_ROOT="${PROS_WORKFLOW_ROOT:-/workspace/pros_workflow}"
export PROS_WORKFLOW_DIR="${PROS_WORKFLOW_DIR:-$PROS_WORKFLOW_ROOT/workflow}"
export PROS_WORKFLOW_PYTHON="${PROS_WORKFLOW_PYTHON:-/opt/pros_workflow_venv/bin/python}"
export PATH="$HOME/.local/bin:$PATH"

run() {
    cd "$PROS_WORKFLOW_DIR" || return 1
    "$PROS_WORKFLOW_PYTHON" main.py --no-mock "$@"
}

web() {
    cd "$PROS_WORKFLOW_DIR" || return 1
    local port="${WEB_PORT:-8080}"
    if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
        port="$1"
        shift
    fi
    "$PROS_WORKFLOW_PYTHON" web_main.py --host "${WEB_HOST:-0.0.0.0}" --port "$port" "$@"
}

if [[ $- == *i* ]]; then
    cd "$PROS_WORKFLOW_ROOT" || :
fi
