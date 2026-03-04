FROM registry.screamtrumpet.csie.ncku.edu.tw/unity_env/pros_rl_image:latest

# Install ros-humble-nav2-msgs (GPG key may be expired in base image; bypass auth)
RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y --allow-unauthenticated ros-humble-nav2-msgs && \
    rm -rf /var/lib/apt/lists/*
