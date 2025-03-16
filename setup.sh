#!/bin/bash

# Color definitions
source utils/colors.sh


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



# Function to extract description from a script file
get_description() {
    local script_file="$1"
    local description=""
    
    # Look for a description line in the format "# Description: text"
    description=$(grep -m 1 "^# Description:" "$script_file" | sed 's/# Description: //')
    
    # If no description found, try to find any comment line
    if [ -z "$description" ]; then
        description=$(grep -m 1 "^#[^!]" "$script_file" | sed 's/^# //')
    fi
    
    # If still no description, use a default one
    if [ -z "$description" ]; then
        script_name=$(basename "$script_file" .sh)
        display_name=$(echo "$script_name" | sed -r 's/(^|_)([a-z])/\U\2/g' | tr '_' ' ')
        description="Run $display_name operation"
    fi
    
    echo "$description"
}


# Function to show dynamic menu based on available scripts
show_menu() {
    print_banner
    
    echo -e "${WHITE}${BOLD}Available operations:${RESET}"
    
    # Find all executable scripts in utils directory
    utils_scripts=()
    script_names=()
    descriptions=()
    counter=1
    
    # Collect scripts, excluding colors.sh which is a utility
    for script in utils/*.sh; do
        if [ "$(basename "$script")" != "colors.sh" ]; then
            utils_scripts+=("$script")
            
            # Format the script name for display by removing .sh and converting to title case
            script_name=$(basename "$script" .sh)
            display_name=$(echo "$script_name" | sed -r 's/(^)([a-z])/\U\2/g' | tr '_' ' ')
            
            script_names+=("$display_name")
            
            # Get the description from the script file
            description=$(get_description "$script")
            descriptions+=("$description")
            
            echo -e "${MAGENTA}$counter)${RESET} ${GREEN}${display_name}${RESET} - ${description}"
            ((counter++))
        fi
    done
    
    # Add exit option
    echo -e "${MAGENTA}$counter)${RESET} ${RED}Exit${RESET}"
    echo
    echo -e "${WHITE}${BOLD}Enter your choice [1-$counter]:${RESET} "
    read -r choice
    
    # Validate choice
    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$counter" ]; then
        if [ "$choice" -eq "$counter" ]; then
            echo -e "${RED}Exiting...${RESET}"
            exit 0
        else
            # Execute the selected script
            source "${utils_scripts[$choice-1]}"
            # Execute the main function from the script (assuming it has the same name as the file without extension)
            script_base=$(basename "${utils_scripts[$choice-1]}" .sh)
            $script_base
            
            # Return to menu after execution
            echo
            echo -e "${CYAN}Press any key to return to the menu...${RESET}"
            read -n 1
            show_menu
        fi
    else
        echo -e "${RED}Invalid option. Please try again.${RESET}"
        sleep 2
        show_menu
    fi
}


# Start the menu
show_menu