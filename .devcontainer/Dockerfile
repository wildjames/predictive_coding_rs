FROM mcr.microsoft.com/devcontainers/rust:latest

USER root
# specify DOCKER_GID to force container docker group id to match host
RUN if [ -n "${DOCKER_GID}" ]; then \
    if ! getent group docker; then \
    groupadd -g ${DOCKER_GID} docker; \
    else \
    groupmod -g ${DOCKER_GID} docker; \
    fi && \
    usermod -aG docker vscode; \
    fi

RUN apt-get update && \
  apt-get install -y \
    libxcursor1 \
    libxcursor-dev \
    libxkbcommon-x11-0 \
    libxkbcommon-dev \
    x11-apps \
    libgl1 \
    libglx-mesa0 \
    mesa-utils \
    protobuf-compiler

# Install NVIDIA Container Toolkit
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg2 \
  && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && apt-get update \
  && apt-get install -y nvidia-container-toolkit \
  && rm -rf /var/lib/apt/lists/*

# Install cargo-llvm-cov
USER vscode
RUN cargo +stable install cargo-llvm-cov --locked
