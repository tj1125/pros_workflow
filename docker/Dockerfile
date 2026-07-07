FROM ghcr.io/screamlab/pros_car_docker_image:latest

ENV PROS_WORKFLOW_VENV=/opt/pros_workflow_venv \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

SHELL ["/bin/bash", "-c"]
WORKDIR /opt/pros_workflow_deps

COPY workflow/pyproject.toml ./

# pros_car_docker_image already includes ROS Humble, Nav2, rosbridge,
# ros2_laser_scan_matcher, numpy, scipy, pybullet, Pillow, PyYAML, urwid,
# OpenCV, and the ROS Python 3.10 stack. Add only the workflow Python deps.
RUN python3 -m venv --system-site-packages "$PROS_WORKFLOW_VENV" && \
    "$PROS_WORKFLOW_VENV/bin/python" -m pip install --no-cache-dir tomli && \
    "$PROS_WORKFLOW_VENV/bin/python" -c "import subprocess, sys, tomli; deps = tomli.load(open('pyproject.toml', 'rb'))['project']['dependencies']; subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', *deps])"

# Suppress the setuptools/easy_install warning emitted by ament_python under colcon --symlink-install.
RUN sed -i 's|^colcon build --symlink-install$|PYTHONWARNINGS="ignore:easy_install command is deprecated" colcon build --symlink-install|' /workspaces/rebuild_colcon.rc

RUN printf '%s\n' \
    '' \
    '# pros_workflow shell helpers. Keep the base image PROS aliases, including r.' \
    'if [ -f /workspace/pros_workflow/scripts/env.sh ]; then' \
    '    source /workspace/pros_workflow/scripts/env.sh' \
    'fi' \
    >> /root/.bashrc

WORKDIR /workspace/pros_workflow

ENTRYPOINT ["/ros_entrypoint.bash"]
CMD ["bash", "-l"]
