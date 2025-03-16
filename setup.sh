#!/bin/bash

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
RESET='\033[0m'

# Banner function
print_banner(){
    # Logo
    printf "
    \e[48;2;0;0;0m                                \e[m
    \e[48;2;0;0;0m                                \e[m
    \e[48;2;0;0;0m                                \e[m
    \e[48;2;0;0;0m                                \e[m
    \e[48;2;0;0;0m           \e[38;2;67;67;67;48;2;0;0;0m▄\e[38;2;211;211;211;48;2;0;0;0m▄\e[38;2;255;255;255;48;2;78;78;78m▄\e[38;2;255;255;255;48;2;128;128;128m▄▄▄▄▄▄▄▄▄▄▄\e[38;2;239;239;239;48;2;19;19;19m▄\e[38;2;43;43;43;48;2;0;0;0m▄\e[48;2;0;0;0m     \e[m
    \e[48;2;0;0;0m          \e[38;2;195;195;195;48;2;60;60;60m▄\e[38;2;255;255;255;48;2;254;254;254m▄\e[48;2;255;255;255m              \e[38;2;164;164;164;48;2;156;156;156m▄\e[48;2;0;0;0m     \e[m
    \e[48;2;0;0;0m         \e[38;2;198;198;198;48;2;65;65;65m▄\e[48;2;255;255;255m   \e[38;2;209;209;209;48;2;255;255;255m▄\e[38;2;2;2;2;48;2;100;100;100m▄\e[48;2;0;0;0m     \e[38;2;14;14;14;48;2;7;7;7m▄\e[38;2;231;231;231;48;2;211;211;211m▄\e[48;2;255;255;255m   \e[38;2;192;192;192;48;2;255;255;255m▄\e[38;2;0;0;0;48;2;62;62;62m▄\e[48;2;0;0;0m     \e[m
    \e[48;2;0;0;0m  \e[38;2;48;48;48;48;2;19;19;19m▄\e[38;2;255;255;255;48;2;192;192;192m▄▄▄▄▄\e[38;2;255;255;255;48;2;197;197;197m▄\e[48;2;255;255;255m    \e[38;2;255;255;255;48;2;228;228;228m▄\e[38;2;255;255;255;48;2;192;192;192m▄▄▄▄▄▄\e[38;2;255;255;255;48;2;241;241;241m▄\e[48;2;255;255;255m   \e[38;2;196;196;196;48;2;255;255;255m▄\e[38;2;0;0;0;48;2;58;58;58m▄\e[48;2;0;0;0m      \e[m
    \e[48;2;0;0;0m  \e[38;2;0;0;0;48;2;6;6;6m▄\e[38;2;14;14;14;48;2;229;229;229m▄\e[38;2;160;160;160;48;2;255;255;255m▄\e[38;2;192;192;192;48;2;255;255;255m▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄\e[38;2;184;184;184;48;2;255;255;255m▄\e[38;2;60;60;60;48;2;249;249;249m▄\e[38;2;0;0;0;48;2;42;42;42m▄\e[48;2;0;0;0m       \e[m
    \e[48;2;0;0;0m      \e[38;2;144;144;144;48;2;45;45;45m▄\e[38;2;255;255;255;48;2;128;128;128m▄▄▄▄\e[38;2;78;78;78;48;2;34;34;34m▄\e[38;2;36;36;36;48;2;0;0;0m▄\e[48;2;0;0;0m                   \e[m
    \e[48;2;0;0;0m      \e[38;2;140;140;140;48;2;193;193;193m▄\e[48;2;255;255;255m              \e[38;2;158;158;158;48;2;255;255;255m▄\e[38;2;0;0;0;48;2;66;66;66m▄\e[48;2;0;0;0m         \e[m
    \e[48;2;0;0;0m      \e[38;2;0;0;0;48;2;17;17;17m▄\e[38;2;3;3;3;48;2;209;209;209m▄\e[38;2;61;61;61;48;2;255;255;255m▄\e[38;2;64;64;64;48;2;255;255;255m▄\e[38;2;107;107;107;48;2;255;255;255m▄\e[38;2;128;128;128;48;2;255;255;255m▄▄▄▄▄▄▄\e[38;2;119;119;119;48;2;255;255;255m▄\e[38;2;57;57;57;48;2;255;255;255m▄\e[38;2;0;0;0;48;2;161;161;161m▄\e[38;2;0;0;0;48;2;6;6;6m▄\e[48;2;0;0;0m          \e[m
    \e[48;2;0;0;0m                                \e[m
    \e[48;2;0;0;0m                                \e[m
    \e[48;2;0;0;0m                                \e[m
    \e[48;2;0;0;0m                                \e[m
    ";
    # Program

 printf "\e[97m
       ┌────────────────────────┐ 
       │  mirena_ws Setup Tool  │ 
       └────────────────────────┘ 
  \e[0m
  ";

}

# Function to install dependencies
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

# Function to clone modules
clone_modules() {
    echo -e "${YELLOW}${BOLD}Cloning all submodules...${RESET}"
    git submodule init
    git submodule update --recursive
    echo -e "${GREEN}${BOLD}✓ All submodules cloned successfully!${RESET}"
    sleep 2
}

# Function to update submodules
update_submodules() {
    echo -e "${YELLOW}${BOLD}Updating all submodules to latest commit...${RESET}"
    git submodule update --remote --merge
    echo -e "${GREEN}${BOLD}✓ All submodules updated successfully!${RESET}"
    sleep 2
}

# Function to build
build_project() {
    echo -e "${YELLOW}${BOLD}Building project...${RESET}"
    source /opt/ros/jazzy/setup.bash
    colcon build
    echo -e "${GREEN}${BOLD}✓ Build completed!${RESET}"
    sleep 2
}

# Main menu
show_menu() {
    print_banner
    echo -e "${WHITE}${BOLD}Please select an option:${RESET}"
    echo -e "${MAGENTA}1)${RESET} ${GREEN}Install dependencies${RESET} - Install ROS2 Jazzy Desktop and Git LFS"
    echo -e "${MAGENTA}2)${RESET} ${GREEN}Clone modules${RESET}      - Clone all repository submodules"
    echo -e "${MAGENTA}3)${RESET} ${GREEN}Update submodules${RESET}  - Update all submodules to latest commit"
    echo -e "${MAGENTA}4)${RESET} ${GREEN}Build project${RESET}      - Run colcon build"
    echo -e "${MAGENTA}5)${RESET} ${RED}Exit${RESET}"
    echo
    echo -e "${WHITE}${BOLD}Enter your choice [1-5]:${RESET} "
    read -r choice

    case $choice in
        1) install_dependencies; show_menu ;;
        2) clone_modules; show_menu ;;
        3) update_submodules; show_menu ;;
        4) build_project; show_menu ;;
        5) echo -e "${RED}Exiting...${RESET}"; exit 0 ;;
        *) echo -e "${RED}Invalid option. Please try again.${RESET}"; sleep 2; show_menu ;;
    esac
}

# Start the script
show_menu
