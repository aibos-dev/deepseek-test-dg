FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG PYTHON_VERSION=3.12.4
ARG PYTHON_MAJOR=3.12

ENV TZ=Africa/Nairobi \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python$PYTHON_MAJOR \
    UV_PROJECT_ENVIRONMENT="/usr/local/"

WORKDIR /tmp/
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get update && apt-get install -y --no-install-recommends \
    git \
    zip \
    wget \
    curl \
    make \
    llvm \
    ffmpeg \
    tzdata \
    tk-dev \
    graphviz \
    xz-utils \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libsqlite3-dev \
    libgl1-mesa-dev \
    libreadline-dev \
    libncurses5-dev \
    libncursesw5-dev \
    build-essential \
    && cd /usr/local/ && wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tar.xz \
    && tar xvf Python-$PYTHON_VERSION.tar.xz \
    && cd /usr/local/Python-$PYTHON_VERSION \
    && ./configure --enable-optimizations \
    && make install \
    && ln -fs /usr/local/bin/python3.12 /usr/bin/python3 \
    && ln -fs /usr/local/bin/python3.12 /usr/bin/python \
    && rm /usr/local/Python-$PYTHON_VERSION.tar.xz \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf \
    /var/lib/apt/lists/* \
    /var/cache/apt/* \
    /usr/local/src/* \
    /tmp/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv && \
    uv --version

# Copy dependency files
COPY . /workspace
WORKDIR /workspace

# Setup user
ARG UID=1001
ARG GID=1001
ARG USERNAME=devuser
ARG GROUPNAME=devgroup
RUN groupadd -g ${GID} ${GROUPNAME} -f && \
    useradd -m -s /bin/bash -u ${UID} -g ${GID} -c "Docker image user" ${USERNAME} && \
    chown -R ${USERNAME}:${GROUPNAME} /opt && \
    chown -R ${USERNAME}:${GROUPNAME} /usr/local && \
    chown -R ${USERNAME}:${GROUPNAME} /workspace

USER ${USERNAME}:${GROUPNAME}

# Create and activate virtual environment, install dependencies
SHELL ["/bin/bash", "-c"]
RUN set -ex && \
    uv venv /workspace/openr1 --python /usr/local/bin/python3.12 && \
    . /workspace/openr1/bin/activate && \
    uv pip install --upgrade pip && \
    uv pip install vllm>=0.7.0 && \
    uv pip install -e /workspace

CMD ["/bin/bash", "-c", "source /workspace/openr1/bin/activate && /bin/bash"]