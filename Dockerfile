FROM registry.screamtrumpet.csie.ncku.edu.tw/unity_env/pros_rl_image:latest

# Install ROS and Python runtime dependencies needed by the control packages.
RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y --allow-unauthenticated \
        python3-pip \
        python3-urwid \
        ros-humble-nav2-msgs && \
    python3 -m pip install --no-cache-dir \
        numpy==2.0.1 \
        scipy==1.14.0 \
        pybullet \
        uv && \
    rm -rf /var/lib/apt/lists/*

# Do not inherit project-specific shell startup behavior from the base image.
ENV BASH_ENV=
ENV ENV=
ENTRYPOINT ["/bin/bash"]
