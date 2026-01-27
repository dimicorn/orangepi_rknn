FROM arm64v8/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get upgrade -y -qq && \
    apt-get install -y --no-install-recommends \
    openssh-client \
    python3 python3-pip python3-dev pybind11-dev \
    cmake curl build-essential wget git zlib1g zlib1g-dev \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    source $HOME/.local/bin/env && \
    uv sync


WORKDIR /home/inference
COPY . .