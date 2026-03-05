.PHONY: lint format format-check typecheck test check clean install demo

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
	uv tool install . --force

demo: demo.tape  ## Record demo GIF from demo.tape
	vhs demo.tape

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
