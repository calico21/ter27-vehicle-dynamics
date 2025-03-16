#!/bin/bash

# Description: Install ROS2 Jazzy Desktop and Git LFS
install_dependencies() {
    echo -e "${YELLOW}${BOLD}Installing dependencies...${RESET}"
    echo -e "${CYAN}>> Installing ROS2 Jazzy Desktop...${RESET}"
    sudo apt update
    sudo apt install -y ros-jazzy-desktop
    echo -e "${CYAN}>> Installing Git LFS...${RESET}"
    sudo apt install -y git-lfs
    git lfs install
    echo -e "${GREEN}${BOLD}✓ Dependencies installed successfully!${RESET}"
    sleep 2
}