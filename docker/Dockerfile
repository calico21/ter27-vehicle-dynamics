FROM osrf/ros:jazzy-desktop-full

ARG USERNAME=mirena
ARG USER_UID=1000
ARG USER_GID=1000


# Delete user if it exists in container (e.g Ubuntu Noble: ubuntu)
RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi
# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

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
USER $USERNAME
WORKDIR /home/$USERNAME

# Setup Acados (MPC SOLVER GENERATOR)
WORKDIR /home/mirena
RUN git clone https://github.com/acados/acados.git
WORKDIR /home/mirena/acados
RUN git checkout 5847c74fe774137a5fb31c9791bf39aa980b8b05
RUN git submodule update --recursive --init
RUN mkdir -p build 
WORKDIR /home/mirena/acados/build
RUN cmake -DACADOS_WITH_QPOASES=ON ..
RUN make install -j4
#Install python interface for acados
RUN pip install -e /home/mirena/acados/interfaces/acados_template
#Set neccesary env vars for acados
ENV ACADOS_SOURCE_DIR=/home/mirena/acados
ENV LD_LIBRARY_PATH=/home/mirena/acados/lib:$LD_LIBRARY_PATH


# Setup Ipopt (Internal Point Solver Runtime)
#WORKDIR /home/mirena
#RUN git clone https://github.com/coin-or/Ipopt.git
#WORKDIR /home/mirena/Ipopt
#RUN mkdir -p build
#WORKDIR /home/mirena/Ipopt/build
#RUN ../configure && make -j4
#RUN make install



# Install Oh My Zsh (non-interactively)
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
# Copy Zshell config
COPY .zshrc /home/$USERNAME/

#SET default Shell for user
ENV SHELL /bin/zsh
SHELL ["/bin/zsh", "-c"]

# Default launch command
ENTRYPOINT ["/bin/zsh"]
