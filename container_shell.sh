source /workspaces/VLM_RL/container_env.sh
_ros_source_base
ros_ws_source >/dev/null 2>&1 || true

PS1='\[\033[0;33m\]⚡\[\033[0m\] \[\033[97;44m\] \u@\h \[\033[0;34m\]\[\033[30;104m\] \w \[\033[0;34m\]\[\033[0m\] '

cd /workspaces/VLM_RL || return 1
