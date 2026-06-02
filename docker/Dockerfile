FROM ghcr.io/screamlab/pros_car_docker_image:latest

ENV VLM_RL_VENV=/opt/vlm_rl_venv \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

SHELL ["/bin/bash", "-c"]
WORKDIR /opt/vlm_rl_deps

COPY workflow/pyproject.toml ./

# pros_car_docker_image already includes ROS Humble, Nav2, rosbridge,
# ros2_laser_scan_matcher, numpy, scipy, pybullet, Pillow, PyYAML, urwid,
# OpenCV, and the ROS Python 3.10 stack. Add only the workflow Python deps.
RUN python3 -m venv --system-site-packages "$VLM_RL_VENV" && \
    "$VLM_RL_VENV/bin/python" -m pip install --no-cache-dir tomli && \
    "$VLM_RL_VENV/bin/python" -c "import subprocess, sys, tomli; deps = tomli.load(open('pyproject.toml', 'rb'))['project']['dependencies']; subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', *deps])"

# Suppress the setuptools/easy_install warning emitted by ament_python under colcon --symlink-install.
RUN sed -i 's|^colcon build --symlink-install$|PYTHONWARNINGS="ignore:easy_install command is deprecated" colcon build --symlink-install|' /workspaces/rebuild_colcon.rc

RUN printf '%s\n' \
    '' \
    '# VLM_RL workflow helpers. Keep the base image PROS aliases, including r.' \
    'if [ -f /workspace/VLM_RL/scripts/env.sh ]; then' \
    '    source /workspace/VLM_RL/scripts/env.sh' \
    'fi' \
    >> /root/.bashrc

WORKDIR /workspace/VLM_RL

ENTRYPOINT ["/ros_entrypoint.bash"]
CMD ["bash", "-l"]
