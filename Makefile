.PHONY: demo demo-prep demo-publish opencode-reel

# Per-model opencode reel. The QA pod emits opencode-<family>.{gif,webm}; this
# renames each to its full model name (demos-src/opencode-model-reel/reel.map)
# and transcodes webm -> mp4 (h264, web-compatible) into demos/.
REEL_INCOMING ?= _qa_demos_incoming
REEL_MAP := demos-src/opencode-model-reel/reel.map

# Demo generation lives here, off main. scripts/demo.sh on main is a
# thin wrapper that worktrees gh-pages and delegates to these targets.
# LILBEE_REPO_ROOT (passed in by the wrapper) points at the main
# checkout so the agent integration recipes + lilbee-mcp skill resolve.

TAPES := tui-setup tui-chat tui-add tui-catalog tui-settings tui-palette \
         tui-crawl tui-tour tui-unsupported mcp-godot-search mcp-godot \
         mcp-manual mcp-code mcp-self-tune

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

opencode-reel:  ## Rename + transcode opencode-<family>.{gif,webm} from $(REEL_INCOMING) into demos/
	@command -v ffmpeg >/dev/null 2>&1 || (echo "ffmpeg required for webm -> mp4" >&2; exit 1)
	@test -d "$(REEL_INCOMING)" || (echo "$(REEL_INCOMING) not found (set REEL_INCOMING=<dir>)" >&2; exit 1)
	@grep -vE '^\s*(#|$$)' $(REEL_MAP) | while read -r fam full; do \
		gif="$(REEL_INCOMING)/opencode-$$fam.gif"; webm="$(REEL_INCOMING)/opencode-$$fam.webm"; \
		[ -f "$$gif" ] || { echo "skip $$fam (no gif)"; continue; }; \
		cp -f "$$gif" "demos/opencode-$$full.gif"; \
		if [ -f "$$webm" ]; then \
			ffmpeg -nostdin -y -loglevel error -i "$$webm" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "demos/opencode-$$full.mp4"; \
		fi; \
		echo "==> $$fam -> $$full"; \
	done

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
