# Quality gates for the gheim repo. Mirrors what CI / pre-commit run.
#
#   make lint      # ruff
#   make typecheck # ty (red on real errors only — ruff catches noise)
#   make test      # pytest (excludes test_data_layers; needs built data)
#   make data-test # full pytest including data-layer integration tests
#   make check     # all of the above (no data-test)
#   make fix       # ruff --fix (auto-applies safe fixes)
#
# All targets are POSIX-make; no GNU extensions required beyond .PHONY.

.PHONY: lint typecheck test data-test check fix

PY_DIRS := training/ packages/ server/

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
