#!/bin/bash

# Description: Install ROS2 Jazzy Desktop and Git LFS
install_dependencies() {
    echo -e "${YELLOW}${BOLD}Installing dependencies...${RESET}"
    
    echo -e "${CYAN}>> Adding ROS2 repository...${RESET}"
    sudo apt update && sudo apt install -y software-properties-common
    sudo add-apt-repository universe
    sudo apt update && sudo apt install -y curl
    
    echo -e "${CYAN}>> Adding ROS 2 GPG key...${RESET}"
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo tee /usr/share/keyrings/ros-archive-keyring.gpg > /dev/null
    
    echo -e "${CYAN}>> Adding ROS 2 Jazzy repository...${RESET}"
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    
    echo -e "${CYAN}>> Installing ROS2 Jazzy Desktop...${RESET}"
    sudo apt update
    sudo apt install -y ros-jazzy-desktop
    
    echo -e "${CYAN}>> Installing Git LFS...${RESET}"
    sudo apt install -y git-lfs
    git lfs install
    
    echo -e "${CYAN}>> Sourcing ROS2...${RESET}"
    echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
    source /opt/ros/jazzy/setup.bash
    
    echo -e "${CYAN}>> Installing rosdep...${RESET}"
    sudo apt install -y python3-rosdep
    sudo rosdep init
    rosdep update
    
    echo -e "${CYAN}>> Installing workspace dependencies...${RESET}"
    if [ -f "src" ]; then
        rosdep install --from-paths src --ignore-src -r -y
    fi
    
    echo -e "${GREEN}${BOLD}✓ Dependencies installed successfully!${RESET}"
    sleep 2
}
