# Contributing to lilbee

Thanks for your interest in contributing!

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- [uv](https://docs.astral.sh/uv/) package manager

## Getting Started

```bash
git clone https://github.com/tobocop2/lilbee.git
cd lilbee
uv sync
```

## Before Submitting

1. Run `make format` to auto-format code
2. Run `make check` to run all checks (lint, format, typecheck, tests)
3. Ensure 100% test coverage — add tests for any new code
4. Keep commits focused and descriptive

## Guidelines

- **Open an issue before large changes** — discuss the approach first
- **100% test coverage** is enforced by CI; PRs that drop coverage will fail
- **No LangChain** — we use the raw Ollama SDK
- Follow existing code style (type hints, dataclasses, small functions)

## Running Tests

```bash
make test       # Run tests with coverage
make lint       # Ruff linting
make typecheck  # Mypy
make check      # All of the above (same as CI)
```
