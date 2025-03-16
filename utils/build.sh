#!/bin/bash

# Description: Run colcon build for the ROS2 project workspace

build() {
    echo -e "${YELLOW}${BOLD}Building project...${RESET}"
    source /opt/ros/jazzy/setup.bash
    colcon build
    echo -e "${GREEN}${BOLD}✓ Build completed!${RESET}"
}