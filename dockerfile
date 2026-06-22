FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

WORKDIR /wrk
RUN apt-get update && apt-get install -y vim cmake parallel curl libssl-dev pkg-config
RUN pip install --no-cache-dir --break-system-packages numpy pandas ipython chess
RUN  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

COPY . /src
WORKDIR  /src
#ENV LIBTORCH=/usr/local/lib/python3.12/dist-packages/torch
#ENV LIBTORCH_INCLUDE=/usr/local/lib/python3.12/dist-packages/torch
ENV LIBTORCH_USE_PYTORCH=1
ENV AOT_INDUCTOR_DEBUG_COMPILE=1
ENV OMP_NUM_THREADS=1
RUN . "$HOME/.cargo/env" && cargo b
