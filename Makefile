.PHONY: lint format format-check typecheck test test-ci test-ci-serial test-ci-forked test-integration imports-check check clean install demo demo-prep demo-publish build publish docs docs-api docs-site site site-serve site-tar dns-setup

lint:
	uv run ruff check src/ tests/ tools/qa/
	uv run python scripts/check_style_rules.py

format:
	uv run ruff format src/ tests/ tools/qa/

format-check:
	uv run ruff format --check src/ tests/ tools/qa/

typecheck:
	uv run mypy src/lilbee/

test:
	uv run pytest --cov=lilbee --cov-report=term-missing -v -n logical --dist loadgroup

test-ci:
	uv run pytest --cov=lilbee --cov-report=term-missing --cov-report=html -v -n logical --dist loadgroup

test-ci-serial:
	uv run pytest --cov=lilbee --cov-report=term-missing --cov-report=html -v -p no:xdist

test-ci-forked:
	uv run pytest --forked -v -n logical --dist loadgroup

imports-check:
	uv run python -c "import lilbee; from lilbee import cli; from lilbee.core import config; from lilbee.data import chunk, code_chunker, store, ingest; from lilbee.retrieval import embedder, query"

test-integration:
	uv run pytest tests/integration/ -v

check: lint format-check typecheck test  ## Run all checks (same as CI)

install:
	uv tool install ".[crawler]" --force --reinstall --compile-bytecode

crawl-setup:  ## Download Playwright Chromium for /crawl
	uv run playwright install chromium

dns-setup:  ## One-time DNS setup for lilbee.sh at Porkbun (reads creds from pass)
	bash scripts/porkbun-dns-setup.sh

demo-prep:  ## Stage demo data dirs via the gh-pages worktree
	bash scripts/demo.sh prep

demo:  ## Render every demo GIF via the gh-pages worktree
	bash scripts/demo.sh render

demo-publish:  ## Commit + push refreshed renders on gh-pages
	bash scripts/demo.sh publish

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

site: docs-api  ## Build the deployable site locally (marketing pages + API docs; no coverage build)
	@echo "site/ is ready. Preview with: make site-serve"

site-serve: site  ## Build + serve the site at http://localhost:8000
	cd site && python3 -m http.server 8000

site-tar: site  ## Build the site and pack it into site.tar.gz
	tar -czf site.tar.gz -C site .
	@echo "wrote site.tar.gz"

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage dist/ openapi.json site/api site/coverage site.tar.gz
	find . -type d -name __pycache__ -exec rm -rf {} +
