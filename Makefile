.PHONY: lint format format-check typecheck test test-ci imports-check check clean install demo build publish plugin-test

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

test-ci:
	uv run pytest --cov=lilbee --cov-report=term-missing --cov-report=html -v

imports-check:
	uv run python -c "import lilbee; from lilbee import cli, config, chunker, code_chunker, embedder, store, ingest, query"

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

plugin-test:
	cd plugins/obsidian && npm test

plugin-build:
	cd plugins/obsidian && npm run build

plugin-dev:
	cd plugins/obsidian && npm run dev

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage dist/
	find . -type d -name __pycache__ -exec rm -rf {} +
