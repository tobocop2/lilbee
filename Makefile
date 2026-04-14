.PHONY: lint format format-check typecheck test test-ci test-integration imports-check check clean install demo build publish docs docs-api docs-site

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

format-check:
	uv run ruff format --check src/ tests/

typecheck:
	uv run mypy src/lilbee/

test:
	uv run pytest --cov=lilbee --cov-report=term-missing -v -n auto

test-ci:
	uv run pytest --cov=lilbee --cov-report=term-missing --cov-report=html -v -n auto

test-ci-serial:
	uv run pytest --cov=lilbee --cov-report=term-missing --cov-report=html -v

imports-check:
	uv run python -c "import lilbee; from lilbee import cli, config, chunk, code_chunker, embedder, store, ingest, query"

test-integration:
	uv run pytest tests/integration/ -v

check: lint format-check typecheck test  ## Run all checks (same as CI)

install:
	uv tool install . --force --reinstall --compile-bytecode

demo:  ## Record all demo GIFs via VHS
	vhs demos/chat.tape
	vhs demos/code-search.tape
	vhs demos/json.tape
	vhs demos/opencode.tape

build:
	uv build

publish: build  ## Build and upload to PyPI
	uv publish

docs-api:  ## Generate OpenAPI schema and Redoc static HTML
	uv run python -c "\
	from lilbee.server.app import create_app; \
	import json; \
	app = create_app(); \
	schema = app.openapi_schema.to_schema(); \
	open('openapi.json', 'w').write(json.dumps(schema, indent=2))"
	npx --yes @redocly/cli build-docs openapi.json -o site/api/index.html
	rm -f openapi.json

docs-site: docs-api  ## Build the full dev portal (coverage + API docs)
	$(MAKE) test-ci
	cp -r htmlcov site/coverage

docs: docs-site  ## Alias for docs-site

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage dist/ openapi.json
	find . -type d -name __pycache__ -exec rm -rf {} +
