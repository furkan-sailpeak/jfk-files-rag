VENV   ?= .venv
PY     := $(VENV)/bin/python
BIN    := $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help install hooks precommit lint test check up down rebuild logs prompt-check

help:  ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

$(VENV)/.installed: requirements-dev.txt rag/requirements.txt
	python3 -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -r requirements-dev.txt
	touch $@

install: $(VENV)/.installed  ## Create .venv with app + dev dependencies

hooks: install  ## Install the git pre-commit hook (runs on every commit)
	$(BIN)/pre-commit install

precommit: install  ## Run all pre-commit checks (ruff, trailing whitespace, EOF, yaml...) on every file
	$(BIN)/pre-commit run --all-files

lint: install  ## Ruff only
	$(BIN)/ruff check rag

test: install  ## Run rag/tests (offline: no LLM, embedding or DB calls)
	$(BIN)/pytest

check: precommit test  ## Pre-commit checks + tests

up:  ## Build and start the app in Docker (http://localhost:8080)
	docker compose up -d --build

down:  ## Stop the Docker app
	docker compose down

rebuild: up  ## Alias for up (rebuilds the image)

logs:  ## Tail the Docker app logs
	docker compose logs -f app

prompt-check:  ## Re-run every LLM stage 3x and report contract/consistency (needs `make up`, uses Groq)
	docker compose exec -T app python - < rag/prompt_check.py
