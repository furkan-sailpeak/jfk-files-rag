# Forward every target to the repo-root Makefile, so `make test` etc. work
# from inside rag/ too. The venv lives at the repo root (../.venv).
.DEFAULT_GOAL := help

%:
	@$(MAKE) --no-print-directory -C .. $@

help:
	@$(MAKE) --no-print-directory -C .. help
