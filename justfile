# Justfile for allocation engine

# Auto-detect uv, venv, or system python
# Use ${REPL_ID:-} to handle unset variable in zsh
PYTHON := `command -v uv >/dev/null 2>&1 && [ -z "${REPL_ID:-}" ] && echo "uv run python" || echo "python"`

# Install tooling
init:
    #!/usr/bin/env bash
    if command -v uv >/dev/null 2>&1; then
        echo "Using uv..."
        uv sync --extra dev
        uv run pre-commit install
    else
        echo "Using pip..."
        python -m pip install -U pip
        pip install -e ".[dev]"
        pre-commit install
    fi

# Run full pipeline (lint + test)
ci: lint test

# Fast tests (~8s)
test:
    @{{PYTHON}} -m pytest tests/unit/ tests/integration/ -q --tb=line --no-cov --log-cli-level=ERROR

# Run tests with coverage (~14s)
test-cov:
    @{{PYTHON}} -m pytest tests/unit/ tests/integration/ --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=90

# E2E stress tests (slow)
test-e2e:
    @{{PYTHON}} -m pytest tests/e2e/ -v --no-cov

# Check code quality
lint:
    @{{PYTHON}} -m ruff check src/ tests/
    @{{PYTHON}} -m black --check src/ tests/

# Format code
format:
    @{{PYTHON}} -m ruff format src/ tests/
    @{{PYTHON}} -m black src/ tests/

# Run dashboard server
dashboard:
    @{{PYTHON}} -m dashboard.run

# Clean build artifacts
clean:
    @rm -rf .coverage htmlcov/ .pytest_cache/ __pycache__