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
