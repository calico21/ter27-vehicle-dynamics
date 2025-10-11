FROM osrf/ros:jazzy-desktop-full

# Install deps
RUN apt update && apt install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    unzip \
    zsh \
    neovim \
    neofetch \
    && rm -rf /var/lib/apt/lists/*
# Use systemwide pip
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# Set workspace
WORKDIR /root/mirena_ws

# Install Oh My Zsh (non-interactively)
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Copy ZSHRC
COPY .zshrc /root/.zshrc

# Prepara el shell de uso
USER root
SHELL ["/bin/zsh", "-c"]

# Default launch command
ENTRYPOINT ["/bin/zsh"]
