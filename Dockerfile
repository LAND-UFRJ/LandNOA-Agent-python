FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ca-certificates curl build-essential \
    libglib2.0-0 libgl1 libglvnd0 libgl1-mesa-dri \
    libsm6 libxext6 libxrender1 libx11-6 \
 && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

COPY requirements.txt .

RUN uv pip install --system -r requirements.txt \
 && uv pip install --system --upgrade --force-reinstall opencv-python-headless
 
COPY . .

RUN uv run initial_config.py

EXPOSE 10000 8501