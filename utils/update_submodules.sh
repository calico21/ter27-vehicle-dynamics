#!/bin/bash

# Description: Update all submodules to latest commit on remote

update_submodules() {
    echo -e "${YELLOW}${BOLD}Updating all submodules to latest commit...${RESET}"
    git submodule update --remote --merge
    echo -e "${GREEN}${BOLD}✓ All submodules updated successfully!${RESET}"
}
