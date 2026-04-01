ARG BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libcairo2-dev \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi8 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=
RUN pip install --upgrade pip \
    && if [ -n "$PIP_EXTRA_INDEX_URL" ]; then \
         pip install -r /app/requirements.txt --index-url "$PIP_INDEX_URL" --extra-index-url "$PIP_EXTRA_INDEX_URL"; \
       else \
         pip install -r /app/requirements.txt --index-url "$PIP_INDEX_URL"; \
       fi

COPY . /app

RUN mkdir -p /app/outputs /app/uploads /app/.cache/huggingface

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
