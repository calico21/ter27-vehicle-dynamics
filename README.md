# Mirena Workspace
Welcome to the mirena project workspace, here you can start off with the development of 
the autopilot system for Tecnun-eRacing TeR Vehicle, you just have to clone the repo, and initialize
all its submodules, there you can build them and launch them using colcon.

# Getting started
Before anything ensure you have godot and ros2-desktop installed and sourced, you can check this by being 
able to run the `ros2` and `godot` commands in shell.
To start with simulator development just init and clone all submodules recursivelly and build with colcon.
```shell
git submodule update --init --remote --recursive
colcon build
source /install/local_setup.sh # Change for your own shell
ros2 launch mirena_sim mirena_sim_launch.py
```

> [!TIP]
> Si fuera fácil, lo haría cualquiera, hazlo.  
> **Ozuba.**
