source /workspaces/VLM_RL/container_env.sh
_ros_source_base
ros_ws_source >/dev/null 2>&1 || true

PS1='\[\033[38;5;220m\]⚡\[\033[0m\] \[\033[97;48;5;238m\] \u@\h \[\033[38;5;27m\]\[\033[48;5;238m\]\[\033[97;48;5;27m\] \w \[\033[38;5;27m\]\[\033[0m\] '

cd /workspaces/VLM_RL || return 1
