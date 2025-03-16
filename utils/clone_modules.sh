#!/bin/bash

# Description: Clone all Git repository submodules recursively

clone_modules() {
    echo -e "${YELLOW}${BOLD}Cloning all submodules...${RESET}"
    git submodule init
    git submodule update --recursive
    echo -e "${GREEN}${BOLD}✓ All submodules cloned successfully!${RESET}"
}