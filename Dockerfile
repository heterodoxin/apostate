ARG USER_ID=1000
ARG GROUP_ID=1000

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS apostate

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NODE_VERSION=18

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    curl \
    git \
    build-essential \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3 \
    && curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

ARG USER_ID
ARG GROUP_ID
RUN groupadd -g ${GROUP_ID} apostate \
    && useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash apostate

WORKDIR /app

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY package.json /app/package.json
RUN npm install --omit=dev

COPY . /app

RUN mkdir -p /home/apostate/.cache/huggingface \
    && chown -R apostate:apostate /app /home/apostate

USER apostate

ENTRYPOINT ["node", "/app/main.js"]


FROM apostate AS apostate-vllm

USER root
RUN python3 -m pip install --no-cache-dir vllm
USER apostate
