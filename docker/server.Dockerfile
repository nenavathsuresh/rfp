FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY agents ./agents
COPY app ./app
COPY client ./client
COPY config ./config
COPY instructions ./instructions
COPY knowledge ./knowledge
COPY models ./models
COPY observability ./observability
COPY tools ./tools
COPY utils ./utils
COPY workflow ./workflow
COPY main.py index.html run_analysis.py ./

RUN python -m pip install --upgrade pip \
    && python -m pip install .

RUN mkdir -p /app/logs /app/runs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
