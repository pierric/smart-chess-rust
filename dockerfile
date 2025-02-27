FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /wrk
RUN apt-get update && apt-get install -y vim cmake parallel curl
RUN pip install --no-cache-dir numpy pandas ipython chess && \
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

COPY . /src
WORKDIR  /src
ENV LIBTORCH=/opt/conda/lib/python3.11/site-packages/torch
ENV LIBTORCH_INCLUDE=/opt/conda/lib/python3.11/site-packages/torch
ENV AOT_INDUCTOR_DEBUG_COMPILE=1
ENV OMP_NUM_THREADS=1
RUN source "$HOME/.cargo/env" && cargo b
