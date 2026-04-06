export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_INSTALL_DIR=/workspaces/VLM_RL/.uv_python
export UV_PROJECT_ENVIRONMENT=/workspaces/VLM_RL/.venv_linux

source /opt/ros/humble/setup.bash
source /workspaces/VLM_RL/container_env.sh

if [ -f /workspaces/install/setup.bash ] && find /workspaces/build -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | grep -q .; then
    source /workspaces/install/setup.bash
fi

cd /workspaces/VLM_RL || return 1
