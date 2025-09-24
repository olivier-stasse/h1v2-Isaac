FROM osrf/ros:jazzy-desktop-full

### INSTALL UV ###
# The installer requires curl (and certificates) to download the release archive
# this step was done when installing the dependencies
# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Clone the repository 
RUN git clone https://github.com/Gepetto/h1v2-Isaac.git
WORKDIR h1v2-Isaac 
RUN git checkout devel

# Synchronize the UV environment
RUN uv sync


