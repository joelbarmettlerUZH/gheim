# Quality gates and developer commands for the gheim repo.
#
#   make help        # this list
#   make lint        # ruff
#   make typecheck   # ty (real errors only)
#   make test        # pytest (excludes data-layer integration tests)
#   make data-test   # full pytest including data-layer tests (needs built data)
#   make check       # lint + typecheck + test
#   make fix         # ruff --fix (auto-apply safe fixes)
#   make clean       # remove caches and build artefacts
#
# All targets are POSIX-make; no GNU extensions required beyond .PHONY.

.PHONY: help lint typecheck test data-test check fix clean

PY_DIRS := training/ packages/ server/

help:
	@echo "Available targets:"
	@echo "  make lint        - ruff check"
	@echo "  make typecheck   - ty type check"
	@echo "  make test        - pytest (no data-layer integration tests)"
	@echo "  make data-test   - pytest including data-layer tests (needs built data)"
	@echo "  make check       - lint + typecheck + test"
	@echo "  make fix         - ruff --fix"
	@echo "  make clean       - remove caches and build artefacts"

lint:
	uvx ruff check $(PY_DIRS)

typecheck:
	uvx ty check training/src/gheim_training/

test:
	uv run pytest training/tests/ --ignore=training/tests/test_data_layers.py \
	    packages/gheim-py/tests/ server/tests/

data-test:
	uv run pytest training/tests/

check: lint typecheck test

fix:
	uvx ruff check $(PY_DIRS) --fix

clean:
	rm -rf .ruff_cache .pytest_cache .mypy_cache
	find . -type d -name '__pycache__' -not -path './.venv/*' -not -path './node_modules/*' -exec rm -rf {} +
	find . -type d -name '*.egg-info' -not -path './.venv/*' -exec rm -rf {} +
