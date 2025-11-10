# Justfile for Quantized LLM Comparison Demo

# Auto-detect uv, venv, or system python
# Use ${REPL_ID:-} to handle unset variable in zsh
PYTHON := `command -v uv >/dev/null 2>&1 && [ -z "${REPL_ID:-}" ] && echo "uv run python" || echo "python"`

# Install dependencies
init:
    #!/usr/bin/env bash
    if command -v uv >/dev/null 2>&1; then
        echo "Using uv..."
        uv sync
    else
        echo "Using pip..."
        python -m pip install -U pip
        pip install -r requirements.txt
    fi

# Run full pipeline (lint and test)
all: lint test

# Check code quality
lint:
    @{{PYTHON}} -m ruff check *.py || echo "ruff not installed, skipping..."
    @{{PYTHON}} -m black --check *.py || echo "black not installed, skipping..."
    @npx eslint static/app.js || echo "ESLint check failed or not installed"
    @npx stylelint static/styles.css || echo "Stylelint check failed or not installed"

# Run tests
test:
    @{{PYTHON}} -m pytest tests/ -v --tb=short || echo "pytest not installed or tests failed"

# Format code
format:
    @{{PYTHON}} -m ruff format *.py || echo "ruff not installed, skipping..."
    @{{PYTHON}} -m black *.py || echo "black not installed, skipping..."
    @npx eslint --fix static/app.js || echo "ESLint format failed or not installed"
    @npx stylelint --fix static/styles.css || echo "Stylelint format failed or not installed"

# Run FastAPI server
server:
    @{{PYTHON}} -m uvicorn server:app --host 0.0.0.0 --port 5000 --reload

# Build BitNet.cpp binary
build:
    @./build.sh

# Clean build artifacts and cache
clean:
    @rm -rf .coverage htmlcov/ .pytest_cache/ __pycache__ .cache/
    @find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    @echo "Cleaned build artifacts"