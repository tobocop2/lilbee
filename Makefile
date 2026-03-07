.PHONY: lint format format-check typecheck test check clean install demo build publish

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

format-check:
	uv run ruff format --check src/ tests/

typecheck:
	uv run mypy src/lilbee/

test:
	uv run pytest --cov=lilbee --cov-report=term-missing -v

check: lint format-check typecheck test  ## Run all checks (same as CI)

install:
	uv tool install . --force --reinstall

demo:  ## Record all demo GIFs via VHS
	vhs demos/chat.tape
	vhs demos/code-search.tape
	vhs demos/json.tape
	vhs demos/opencode.tape

build:
	uv build

publish: build  ## Build and upload to PyPI
	uv publish

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage dist/
	find . -type d -name __pycache__ -exec rm -rf {} +
