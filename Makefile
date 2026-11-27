.PHONY: coverage-gate lint license-check lint-shell test-shell format format-check typecheck test test-ci test-ci-split test-ci-serial test-ci-forked test-integration test-presets imports-check fuzz-smoke engine-archs docs-settings docs-formats check clean install demo demo-prep demo-publish build publish release promote release-promote docs docs-api docs-site site site-serve site-tar dns-setup qa-pod-volume qa-pod-up qa-pod-logs qa-pod-down

lint: lint-shell
	uv run ruff check src/ tests/ tools/qa/ scripts/qa/ tools/readme_media.py tools/gen_settings_reference.py tools/gen_formats_table.py tools/gen_engine_packs.py tools/build_wheel_indexes.py tools/check_pack_licences.py tools/wheel-build/metal_floor.py tools/wheel-build/llama_cuda_notice.py tools/wheel-build/repack_llama_archives.py hatch_build.py
	uv run python scripts/check_style_rules.py
	# docs/settings.md is generated from the Config fields; a new setting must
	# land in the reference with the code that adds it.
	uv run python tools/gen_settings_reference.py --check
	# The README formats table is generated from discovery's format map, so an
	# xberg or tree-sitter bump that changes the set lands in the table.
	uv run python tools/gen_formats_table.py --check

license-check:  ## fail on a GPL or AGPL package in uv.lock (needs the network and Python 3.12)
	bash tools/ci/check_licenses.sh

lint-shell:  ## shellcheck the release scripts and actionlint every workflow
	@if command -v shellcheck >/dev/null; then \
	  shellcheck -s bash scripts/*.sh tools/ci/*.sh tools/qa/artifact_smoke.sh tools/qa/find_engine_bin.sh tools/wheel-build/pack_*.sh tools/wheel-build/check_*_wheel.sh tools/wheel-build/check_ik_repro.sh tests/shell/*.bash tests/shell/stubs/*; \
	  shellcheck -x -s sh site/install.sh tests/installer/select_asset.sh tests/installer/probe_hardware.sh; \
	else echo "lint-shell: shellcheck is not installed, skipping"; fi
	# actionlint also shellchecks every inline `run:` block.
	@if command -v actionlint >/dev/null; then actionlint; \
	else echo "lint-shell: actionlint is not installed, skipping"; fi
	@if command -v pwsh >/dev/null; then pwsh -NoProfile -File tests/installer/lint.ps1; \
	else echo "lint-shell: pwsh is not installed, skipping PSScriptAnalyzer"; fi

test-shell:  ## Run the bats and Pester tests for the release scripts
	@command -v bats >/dev/null || { echo "test-shell: bats-core is missing (brew install bats-core)"; exit 1; }
	bats tests/shell/
	@if command -v pwsh >/dev/null; then \
	  pwsh -NoProfile -Command "Import-Module Pester -MinimumVersion 5.0; Invoke-Pester -Path tests/installer/install.Tests.ps1 -Output Detailed -CI"; \
	else echo "test-shell: pwsh is not installed, skipping Pester"; fi

format:
	uv run ruff format src/ tests/ tools/qa/ scripts/qa/ tools/readme_media.py tools/gen_settings_reference.py tools/gen_formats_table.py tools/gen_engine_packs.py tools/build_wheel_indexes.py tools/check_pack_licences.py tools/wheel-build/llama_cuda_notice.py tools/wheel-build/repack_llama_archives.py hatch_build.py

format-check:
	uv run ruff format --check src/ tests/ tools/qa/ scripts/qa/ tools/readme_media.py tools/gen_settings_reference.py tools/gen_formats_table.py tools/gen_engine_packs.py tools/build_wheel_indexes.py tools/check_pack_licences.py tools/wheel-build/llama_cuda_notice.py tools/wheel-build/repack_llama_archives.py hatch_build.py

typecheck:
	uv run mypy src/lilbee/
	# The GPU QA tools read lilbee's own types, so they rot silently when one
	# changes: multi_gpu_smoke crashed on its first check from #540 until this
	# was added. The rest of tools/qa/ is not clean yet and is not listed.
	uv run mypy tools/qa/placement_matrix/ tools/qa/engine_demo/ tools/qa/multi_gpu_smoke.py
	# hatch_build.py builds the PyPI long description, so a break here breaks
	# the release rather than a test.
	uv run mypy tools/readme_media.py tools/gen_formats_table.py tools/gen_engine_packs.py tools/build_wheel_indexes.py tools/check_pack_licences.py tools/wheel-build/llama_cuda_notice.py tools/wheel-build/repack_llama_archives.py hatch_build.py

# A replaced worker deadlocks --dist loadgroup (pytest-dev/pytest-xdist#1378),
# so a worker that dies fails the run instead.
XDIST := -n logical --dist loadgroup --max-worker-restart=0

# The pack tests read the build workflow and the cells file of the engine-packs repository.
# A sibling checkout named lilbee-engine-packs next to this one supplies them.
ENGINE_PACKS_WORKFLOW ?= $(abspath $(wildcard ../lilbee-engine-packs/.github/workflows/build-packs.yml))
export ENGINE_PACKS_WORKFLOW
ENGINE_PACKS_CELLS ?= $(abspath $(wildcard ../lilbee-engine-packs/cells.yml))
export ENGINE_PACKS_CELLS

# The hang watchdog (tests/_hang_watchdog.py) prints every stack when a test runs long,
# so a lost worker leaves a trace in the job log. This Makefile owns the seconds for
# the unit test recipes, and CI sets only the folder. An empty value would switch the
# watchdog off without a word, so make stops instead. The integration and preset
# recipes do not get the variables: the watchdog ignores a test's own timeout, so it
# would kill a slow test that is marked to run longer.
LILBEE_TEST_HANG_DUMP_S ?= 90
LILBEE_TEST_HANG_DUMP_DIR ?= $(or $(TMPDIR),/tmp)/lilbee-hang-dumps
ifeq ($(strip $(LILBEE_TEST_HANG_DUMP_S)),)
$(error LILBEE_TEST_HANG_DUMP_S is empty: the hang watchdog would be off)
endif
UNIT_TEST_RECIPES := test test-ci test-ci-split test-ci-serial test-ci-forked
$(UNIT_TEST_RECIPES): export LILBEE_TEST_HANG_DUMP_S := $(LILBEE_TEST_HANG_DUMP_S)
$(UNIT_TEST_RECIPES): export LILBEE_TEST_HANG_DUMP_DIR := $(LILBEE_TEST_HANG_DUMP_DIR)

# Branch data is on for the whole run. coverage.py has no per-package branch switch, so pytest-cov's
# own threshold is off and two reports of the one .coverage file gate at 100%: one counts branches
# outside LINE_ONLY, the other counts lines only inside it (.coveragerc-lineonly). Each step that
# clears a package takes it out of LINE_ONLY; when it is empty, set branch = true in pyproject.toml
# and drop both reports. The Windows lane and the site build measure lines only: they set COV_FLAGS
# and BRANCH_CHECK themselves.
LINE_ONLY := src/lilbee/cli/tui/*,src/lilbee/providers/fleet/*
COV_FLAGS ?= --cov-branch --cov-fail-under=0
BRANCH_CHECK ?= $(MAKE) --no-print-directory coverage-gate

coverage-gate:
	uv run coverage report --include='src/lilbee/*' --omit='src/lilbee/_generated/*,$(LINE_ONLY)' --fail-under=100 --skip-covered
	uv run coverage report --rcfile=.coveragerc-lineonly --include='$(LINE_ONLY)' --fail-under=100 --skip-covered

test:
	uv run pytest --cov=lilbee --cov-report=term-missing $(COV_FLAGS) -v $(XDIST)
	$(BRANCH_CHECK)

test-ci:
	uv run pytest --cov=lilbee --cov-report=term-missing --cov-report=html $(COV_FLAGS) -v $(XDIST)
	$(BRANCH_CHECK)

# One coverage session over the whole suite stalls the TUI pilot past its 30 s
# screen wait on the 3 vCPU macOS runner; two shorter sessions do not.
TUI_TESTS := tests/test_tui*.py

test-ci-split:
	uv run pytest --cov=lilbee --cov-report= --cov-fail-under=0 -v $(XDIST) $(wildcard $(TUI_TESTS))
	uv run pytest --cov=lilbee --cov-append --cov-report=term-missing --cov-report=html -v $(XDIST) '--ignore-glob=$(TUI_TESTS)'

test-ci-serial:
	uv run pytest --cov=lilbee --cov-report=term-missing --cov-report=html -v -p no:xdist

test-ci-forked:
	uv run pytest --forked -v $(XDIST)

imports-check:
	uv run python -c "import lilbee; from lilbee import cli; from lilbee.core import config; from lilbee.data.extract import chunk, code_chunker; from lilbee.data import store, ingest; from lilbee.retrieval import embedder, query"

# The hosted-key tests need the providers extra, which only this job installs.
# Every test that needs the library lives in tests/integration/ and imports it plainly,
# so a job without the extra fails here and the plain test job never collects them.
test-integration:
	uv run pytest tests/integration/ tests/providers/test_hosted_request_key.py tests/test_estimator_real_binary.py tests/test_gguf_meta.py -v

# The preset replay needs the client library, so the plain test job cannot collect it.
# It is offline and takes about a minute, so the local gate runs it by name.
test-presets:
	uv run pytest tests/integration/test_preset_fixtures.py -v $(XDIST)

fuzz-smoke:  ## Seeded adversarial TUI fuzz, fixed seeds (deterministic, CI-sized)
	uv run python scripts/qa/tui_fuzz.py smoke

check: lint license-check format-check typecheck test test-presets test-shell  ## Run all checks (same as CI)

install:
	uv tool install ".[crawler]" --force --reinstall --compile-bytecode

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

release:  ## Bump the beta version, tag, and push; CI builds + publishes
	bash scripts/release.sh

promote:  ## Re-release an existing tag's source under a new version (FROM=v0.6.90b420.dev719 TO=0.6.90b420; DRY_RUN=1 to preview)
	@[ -n "$(FROM)" ] && [ -n "$(TO)" ] || { echo "promote: FROM=v... TO=<new-version> required"; exit 1; }
	bash scripts/promote_release.sh $(FROM) $(TO)

release-promote:  ## Manual promote: rewrite notes as headings and mark a release latest (TAG=... or newest). verify-release.yml promotes on its own; use this to rewrite notes or to promote a tag it never ran for
	@tag="$(TAG)"; \
	[ -n "$$tag" ] || tag=$$(gh release list --repo tobocop2/lilbee --limit 30 --json tagName -q "first(.[].tagName | select(startswith(\"v\")))"); \
	verified=$$(gh run list --repo tobocop2/lilbee --workflow=verify-release.yml --limit 50 --json displayTitle,conclusion -q "[.[] | select(.displayTitle == \"Verify release $$tag\" and .conclusion == \"success\")] | length"); \
	if [ "$$verified" = "0" ]; then \
	  echo "REFUSING to promote $$tag: no green 'Verify release' run for it."; \
	  echo "The verify-release workflow must pass against the release assets first:"; \
	  echo "  gh workflow run verify-release.yml -f tag=$$tag"; \
	  exit 1; \
	fi; \
	bash scripts/write_release_sums.sh "$$tag" tobocop2/lilbee; \
	if gh release view "$$tag" --repo tobocop2/lilbee --json body -q .body | head -1 | grep -q '^Promoted from'; then \
	  echo "release-promote: $$tag (verified; promoted release, keeping its copied notes)"; \
	  gh release edit "$$tag" --repo tobocop2/lilbee --prerelease=false --latest; \
	else \
	  prev=$$(gh release list --repo tobocop2/lilbee --exclude-drafts --limit 30 --json tagName -q "first(.[].tagName | select(startswith(\"v\") and . != \"$$tag\"))"); \
	  echo "release-promote: $$tag (verified; notes diff from $$prev)"; \
	  notes=$$(mktemp); \
	  bash scripts/release_notes.sh tobocop2/lilbee "$$tag" "$$prev" > "$$notes"; \
	  gh release edit "$$tag" --repo tobocop2/lilbee --notes-file "$$notes" --prerelease=false --latest; \
	  rm -f "$$notes"; \
	fi

engine-archs:  ## Regenerate the supported-architecture list from the pinned engine
	uv run python tools/gen_supported_archs.py

docs-settings:  ## Regenerate the cross-surface settings reference
	uv run python tools/gen_settings_reference.py

docs-formats:  ## Regenerate the README supported-formats table
	uv run python tools/gen_formats_table.py

docs-api:  ## Generate OpenAPI schema and Redoc static HTML
	uv run python tools/gen_openapi.py openapi.json
	npx --yes @redocly/cli build-docs openapi.json -o site/api/index.html
	rm -f openapi.json

docs-site: docs-api  ## Build the full dev portal (coverage + API docs)
	# Tests gate CI and releases, not the site build; the coverage report deploys regardless of the test outcome.
	# The site shows one percentage, so this run measures lines only.
	-$(MAKE) test-ci COV_FLAGS= BRANCH_CHECK=true
	cp -r htmlcov site/coverage
	uv run python tools/update_site_coverage.py

docs: docs-site  ## Alias for docs-site

site: docs-api  ## Build the deployable site locally (marketing pages + API docs; no coverage build)
	@echo "site/ is ready. Preview with: make site-serve"

site-serve: site  ## Build + serve the site at http://localhost:8000
	cd site && python3 -m http.server 8000

site-tar: site  ## Build the site and pack it into site.tar.gz
	tar -czf site.tar.gz -C site .
	@echo "wrote site.tar.gz"

# --- opencode QA pod (SkyPilot + RunPod) ------------------------------------
# Provisioning is SkyPilot; it reads the RunPod key from ~/.runpod/config.toml.
# One-time: pip install "skypilot[runpod]" && runpod config && sky check runpod.
# See tools/qa/opencode/README.md.
QA_SKY := tools/qa/opencode

qa-pod-volume:  ## Create/adopt the reusable QA network volume (once)
	sky volumes apply $(QA_SKY)/qa-volume.sky.yaml

qa-pod-up:  ## Provision the QA pod, bootstrap, and run the matrix (MATRIX_ARGS=... to narrow)
	# -i 30 --down: autostop after 30 min idle and tear the pod down when the matrix
	# finishes, so a finished or hung run never bills idle. The volume keeps the
	# engine, models, and results; reels just re-launch (a warm volume boots fast).
	# HF_TOKEN (from your shell) is passed as a --secret so gated pulls work and
	# the token stays redacted in sky logs; unset = ungated models only.
	sky launch -c lilbee-qa $(QA_SKY)/qa-pod.sky.yaml -y -i 30 --down $(if $(HF_TOKEN),--secret HF_TOKEN="$(HF_TOKEN)",) $(if $(MATRIX_ARGS),--env MATRIX_ARGS="$(MATRIX_ARGS)",)

qa-pod-logs:  ## Follow the QA pod's run log
	sky logs lilbee-qa

qa-pod-down:  ## Tear down the QA pod (the volume and its state survive)
	sky down lilbee-qa -y

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage dist/ openapi.json site/api site/coverage site.tar.gz
	find . -type d -name __pycache__ -exec rm -rf {} +
