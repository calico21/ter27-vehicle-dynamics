FROM osrf/ros:jazzy-desktop-full

# Install deps
RUN apt update && apt install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    unzip \
    && rm -rf /var/lib/apt/lists/*
# Use systemwide pip
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# Set workspace
WORKDIR /mirena_ws
#COPY . .

# Prepara el shell de uso
USER root
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

# Default launch command
ENTRYPOINT ["/bin/bash"]
