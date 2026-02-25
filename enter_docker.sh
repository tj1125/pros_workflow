#!/bin/bash
# Enter VLM-RL container using the ROS-enabled pros_rl_image
# Mounts the project directory so code changes take effect immediately

docker run -it --rm \
  -v "$(pwd):/workspaces/VLM_RL" \
  --network compose_cube_bridge_network \
  --env-file ./.env \
  --env OLLAMA_URL="http://140.116.82.233:11434" \
  registry.screamtrumpet.csie.ncku.edu.tw/unity_env/pros_rl_image:latest /bin/bash
