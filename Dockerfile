FROM arm64v8/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get upgrade -y -qq && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev pybind11-dev \
    cmake curl build-essential git zlib1g zlib1g-dev \
    libgl1 libglib2.0-0 && \
    pip3 install rknn-toolkit2 torchvision && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/inference
COPY . .