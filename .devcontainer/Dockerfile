FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04
 
# Build arguments
ARG PYTHON_VERSION=3.11.11
ARG NODE_VERSION=20.x
ARG NPM_VERSION=11.0.0
ARG USERNAME=devuser
ARG USER_UID=1001
ARG USER_GID=$USER_UID
 
ENV TZ=Africa/Nairobi \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash \
    POETRY_HOME=/opt/poetry \
    POETRY_VERSION=1.7.1 \
    PATH="/opt/poetry/bin:$PATH" \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 \
    PATH="${PATH}:/usr/lib/jvm/java-17-openjdk-arm64/bin"
 
WORKDIR /tmp/
 
# Install base packages with error handling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    git \
    zip \
    curl \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    nano \
    git-lfs \
    unzip \
    cmake \
    bash \
    openjdk-17-jdk \
    maven \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*
 
# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
 
# Install Python from source with better error handling
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz && \
    tar xf Python-${PYTHON_VERSION}.tar.xz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure \
        --enable-optimizations \
        --with-ensurepip=install \
        --enable-loadable-sqlite-extensions \
        --with-system-ffi \
        --with-computed-gotos \
        --enable-ipv6 && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf Python-${PYTHON_VERSION}* && \
    ln -sf /usr/local/bin/python3 /usr/bin/python3 && \
    ln -sf /usr/local/bin/python3 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3 /usr/bin/pip && \
    python3 -m pip install --upgrade pip
 
# Install Node.js with specific npm version
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION} | bash - && \
    apt-get update && \
    apt-get install -y nodejs && \
    npm install -g npm@${NPM_VERSION}
 
# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    chmod 755 /opt/poetry/bin/poetry && \
    poetry --version
 
# Setup user with explicit shell
RUN groupadd --gid $USER_GID $USERNAME || true && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash || true && \
    mkdir -p /workspace /home/$USERNAME/.cache /home/$USERNAME/.local && \
    chown -R $USERNAME:$USERNAME /workspace /home/$USERNAME && \
    chmod -R 755 /home/$USERNAME && \
    mkdir -p /usr/local/lib/node_modules && \
    chown -R $USERNAME:$USERNAME /usr/local/lib/node_modules && \
    chown -R ${USERNAME}:${USERNAME} /opt/poetry && \
    chown -R $USERNAME:$USERNAME /usr/local/bin && \
    mkdir -p /home/$USERNAME/.npm && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/.npm && \
    mkdir -p /usr/local/share/npm-global && \
    chown -R $USERNAME:$USERNAME /usr/local/share/npm-global
 
# Set npm configuration for global installations
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH="/usr/local/share/npm-global/bin:$PATH"
 
# Create necessary directories for npm-global binaries
RUN mkdir -p /usr/local/share/npm-global/bin && \
    chown -R $USERNAME:$USERNAME /usr/local/share/npm-global/bin
 
# Set up Poetry configuration for devuser
USER $USERNAME
RUN poetry config virtualenvs.create false && \
    mkdir -p /home/$USERNAME/.config/pypoetry && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/.config
 
# Create bash profile for the user
RUN echo 'export PS1="\[\e[32m\]\u@\h:\[\e[34m\]\w\[\e[0m\]\$ "' >> /home/$USERNAME/.bashrc && \
    echo 'alias ll="ls -la"' >> /home/$USERNAME/.bashrc && \
    echo 'alias python=python3' >> /home/$USERNAME/.bashrc && \
    echo 'export PATH="/opt/poetry/bin:$PATH"' >> /home/$USERNAME/.bashrc && \
    echo 'export NODE_PATH=/usr/lib/node_modules' >> /home/$USERNAME/.bashrc && \
    echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64' >> /home/$USERNAME/.bashrc && \
    echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> /home/$USERNAME/.bashrc
 
# Cleanup
RUN apt-get clean || true && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* || true
 
WORKDIR /workspace
 
USER $USERNAME
 
SHELL ["/bin/bash", "-c"]
 
CMD ["/bin/bash"]