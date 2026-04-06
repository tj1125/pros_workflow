FROM registry.screamtrumpet.csie.ncku.edu.tw/unity_env/pros_rl_image:latest

# Install ROS and Python runtime dependencies needed by the control packages.
RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y --allow-unauthenticated \
        python3-pip \
        python3-scipy \
        python3-urwid \
        ros-humble-nav2-msgs && \
    python3 -m pip install --no-cache-dir pybullet && \
    rm -rf /var/lib/apt/lists/*
