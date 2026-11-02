# MT Trading Swarm: developer shortcuts. Run `make` or `make help` for the list.
#
# Variables can be overridden on the command line, e.g.
#   make api API_PORT=9000
#   make backtest BACKTEST_ARGS="--provider yahoo --symbol AAPL --start 2023-01-02 --end 2024-12-31"

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
COMPOSE ?= docker compose
ENV_FILE ?= .env

API_HOST ?= 127.0.0.1
API_PORT ?= 8000
DASHBOARD_PORT ?= 8501
COV_FAIL_UNDER ?= 85
BACKTEST_ARGS ?= --provider synthetic --symbol BULL --symbol FLAT \
	--start 2025-01-02 --end 2026-06-30 --output data/backtest.json

# The API reads .env itself (pydantic-settings), but the dashboard and the CLI only see
# real environment variables. When .env exists, they are started through python-dotenv
# so local runs share one configuration file. Variables already in the shell win.
WITH_ENV = $(if $(wildcard $(ENV_FILE)),$(PYTHON) -m dotenv -f $(ENV_FILE) run --no-override --,)

.DEFAULT_GOAL := help

.PHONY: help install dev env test cov lint format check api dashboard backtest \
	docker-build docker-up docker-down docker-logs clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\nTargets:\n"} \
		/^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the pinned runtime dependencies
	$(PIP) install -r requirements.txt

dev: ## Install runtime + dev tooling and the package in editable mode (adds mt-swarm-backtest)
	$(PIP) install -r requirements-dev.txt
	$(PIP) install --no-deps -e .

env: ## Create .env from .env.example (never overwrites an existing .env)
	@if [ -f $(ENV_FILE) ]; then echo "$(ENV_FILE) already exists; not overwriting"; \
	else cp .env.example $(ENV_FILE) && echo "created $(ENV_FILE) from .env.example"; fi

test: ## Run the test suite (offline, under a minute)
	$(PYTHON) -m pytest -q

cov: ## Run the tests with a coverage report (fails under COV_FAIL_UNDER %)
	$(PYTHON) -m pytest -q --cov=trading_swarm --cov=dashboard --cov=scripts \
		--cov-report=term --cov-report=html --cov-fail-under=$(COV_FAIL_UNDER)

lint: ## ruff check + ruff format --check (what CI runs)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Auto-fix lint findings and format the code
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

check: lint test ## Lint, then test (the full local quality gate)

api: ## Run the API with auto-reload (default http://127.0.0.1:8000, docs at /docs)
	$(PYTHON) -m uvicorn trading_swarm.api.main:app --reload --no-access-log \
		--host $(API_HOST) --port $(API_PORT)

dashboard: ## Run the Streamlit dashboard (default http://localhost:8501)
	$(WITH_ENV) $(PYTHON) -m streamlit run dashboard/app.py --server.port $(DASHBOARD_PORT)

backtest: ## Run a backtest (default: offline synthetic BULL + FLAT, JSON in data/)
	$(PYTHON) -m scripts.run_backtest $(BACKTEST_ARGS)

docker-build: ## Build both Docker images (api, dashboard)
	$(COMPOSE) build

docker-up: ## Build and start the API + dashboard in the background
	$(COMPOSE) up --build -d
	@echo "API:       http://localhost:$(API_PORT)/docs"
	@echo "Dashboard: http://localhost:$(DASHBOARD_PORT)"

docker-down: ## Stop the containers (the portfolio volume is kept)
	$(COMPOSE) down

docker-logs: ## Follow the container logs
	$(COMPOSE) logs -f

clean: ## Remove caches and coverage output (keeps data/ and .env)
	find . -type d -name __pycache__ -not -path "./.venv/*" -not -path "./venv/*" -prune \
		-exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov build dist mt_trading_swarm.egg-info
	rm -f .coverage coverage.xml
