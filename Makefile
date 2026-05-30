.PHONY: demo demo-prep demo-publish

# Demo generation lives here, off main. scripts/demo.sh on main is a
# thin wrapper that worktrees gh-pages and delegates to these targets.
# LILBEE_REPO_ROOT (passed in by the wrapper) points at the main
# checkout so the agent integration recipes + lilbee-mcp skill resolve.

TAPES := tui-setup tui-chat tui-add tui-catalog tui-settings tui-palette \
         tui-crawl tui-tour tui-unsupported mcp-godot-search mcp-godot \
         mcp-manual mcp-code mcp-self-tune tui-ollama-document

demo-prep:  ## Pre-stage models, indexed corpora, opencode demo dirs
	@test -n "$$LILBEE_REPO_ROOT" || (echo "LILBEE_REPO_ROOT must be set (path to main checkout)" >&2; exit 1)
	bash demos-src/_prep.sh

demo:  ## Render every tape in demos-src/ to demos-src/_out/
	@test -n "$$LILBEE_REPO_ROOT" || (echo "LILBEE_REPO_ROOT must be set" >&2; exit 1)
	mkdir -p demos-src/_out
	@for tape in $(TAPES); do \
		echo "==> rendering $$tape"; \
		( cd demos-src && vhs "$$tape.tape" ) || exit 1; \
	done
	@command -v gifsicle >/dev/null 2>&1 && \
		for f in demos-src/_out/*.gif; do gifsicle -O3 --lossy=80 -b "$$f"; done || \
		echo "(install gifsicle to shrink GIFs further)"

demo-publish:  ## Copy rendered GIFs/PNGs into demos/ (served path) + commit
	@test -d demos-src/_out || (echo "run \`make demo\` first" >&2; exit 1)
	cp -f demos-src/_out/*.gif demos/
	cp -f demos-src/_out/*.png demos/
	git add demos/
	@if git diff --cached --quiet; then \
		echo "==> no changes to publish."; \
	else \
		git commit -m "demos: refresh rendered reel"; \
		echo "==> committed on gh-pages. push with: git push origin gh-pages"; \
	fi
