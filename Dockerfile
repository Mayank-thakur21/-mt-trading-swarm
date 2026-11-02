# syntax=docker/dockerfile:1
#
# MT Trading Swarm: one Dockerfile, two runtime images.
#
#   docker build --target api       -t mt-trading-swarm-api .        # FastAPI + scheduler
#   docker build --target dashboard -t mt-trading-swarm-dashboard .  # Streamlit UI
#
# `docker compose up --build` builds both (see docker-compose.yml). A build without
# --target produces the API image, because `api` is the last stage.
#
# Design notes
# * requirements.txt stays the single source of truth for versions. Each image installs
#   only the pins it needs: the API image leaves out streamlit/plotly, and the dashboard
#   image gets streamlit, plotly, pandas, numpy and requests only. The dashboard talks
#   to the API over HTTP and never imports the trading engine.
# * Dependencies go into /opt/venv in a builder stage, which is then copied into a clean
#   runtime stage. Application code stays owned by root, so the unprivileged `app` user
#   can read it but not change it. Only /app/data (the SQLite portfolio) is writable.
# * Health checks use the Python standard library (urllib), so neither image needs curl.

ARG PYTHON_VERSION=3.13

# =============================================================================== base
FROM python:${PYTHON_VERSION}-slim AS base

ARG VERSION=2.0.0
LABEL org.opencontainers.image.title="MT Trading Swarm" \
      org.opencontainers.image.description="Multi-agent paper-trading swarm: trend + Claude news sentiment + risk gate + execution" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/Mayank-thakur21/-mt-trading-swarm"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1

# A fixed, unprivileged uid/gid makes volume ownership predictable across rebuilds.
# `USER 10001:10001` below is this user; the numeric form lets runtimes such as
# Kubernetes (runAsNonRoot) verify it. The home directory is needed for user-level
# caches (yfinance keeps its time-zone cache under ~/.cache).
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app \
       --shell /usr/sbin/nologin app

WORKDIR /app

# =============================================================================== builders
FROM base AS api-deps
COPY requirements.txt ./
# Everything except the dashboard-only libraries. The engine never imports them.
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -vE '^(streamlit|plotly)==' requirements.txt > /tmp/requirements-api.txt \
    && python -m venv /opt/venv \
    && /opt/venv/bin/pip install -r /tmp/requirements-api.txt

FROM base AS dashboard-deps
COPY requirements.txt ./
# The dashboard is a thin HTTP client: streamlit, plotly, pandas, numpy and requests are
# all it imports. The count check stops the build if a pin is renamed or removed.
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -E '^(streamlit|plotly|pandas|numpy|requests)==' requirements.txt \
       > /tmp/requirements-dashboard.txt \
    && test "$(wc -l < /tmp/requirements-dashboard.txt)" -eq 5 \
    && python -m venv /opt/venv \
    && /opt/venv/bin/pip install -r /tmp/requirements-dashboard.txt

# =============================================================================== dashboard
FROM base AS dashboard

COPY --from=dashboard-deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    API_URL=http://api:8000 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY dashboard/ ./dashboard/
# Project-level Streamlit config (theme, hidden error details). Streamlit reads it from
# the working directory.
COPY .streamlit/config.toml ./.streamlit/config.toml

USER 10001:10001
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"]

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true", \
     "--server.fileWatcherType", "none"]

# =============================================================================== api
FROM base AS api

COPY --from=api-deps /opt/venv /opt/venv
# The SQLite file lives on the /app/data volume (see docker-compose.yml).
ENV PATH="/opt/venv/bin:${PATH}" \
    DATABASE_URL=sqlite:////app/data/trading_swarm.db

COPY trading_swarm/ ./trading_swarm/
# The backtest CLI: `docker compose run --rm api python -m scripts.run_backtest ...`
COPY scripts/ ./scripts/

# Create the data directory and give it to `app` before the volume is mounted. On first
# use Docker copies this ownership into a new named volume, so the non-root process can
# write the database.
RUN mkdir -p /app/data && chown app:app /app/data

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# Exactly one worker, on purpose. The paper broker's locks, the rate limiter and the
# scheduler thread are all in-process state, so a second worker would get its own rate
# limits and run a second scheduler. The API logs requests itself (with request ids),
# so uvicorn's own access log is turned off to avoid duplicate lines. The graceful
# timeout lets a running watchlist scan finish before the broker closes.
CMD ["uvicorn", "trading_swarm.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--no-access-log", \
     "--timeout-graceful-shutdown", "25"]
