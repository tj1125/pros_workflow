export VLM_RL_ROOT="${VLM_RL_ROOT:-/workspace/VLM_RL}"
export VLM_RL_WORKFLOW_ROOT="${VLM_RL_WORKFLOW_ROOT:-$VLM_RL_ROOT/workflow}"
export VLM_RL_PYTHON="${VLM_RL_PYTHON:-/opt/vlm_rl_venv/bin/python}"
export PATH="$HOME/.local/bin:$PATH"

run() {
    cd "$VLM_RL_WORKFLOW_ROOT" || return 1
    "$VLM_RL_PYTHON" main.py --no-mock "$@"
}

web() {
    cd "$VLM_RL_WORKFLOW_ROOT" || return 1
    local port="${WEB_PORT:-8080}"
    if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
        port="$1"
        shift
    fi
    "$VLM_RL_PYTHON" web_main.py --host "${WEB_HOST:-0.0.0.0}" --port "$port" "$@"
}

if [[ $- == *i* ]]; then
    cd "$VLM_RL_ROOT" || :
fi
