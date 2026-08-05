Hide
Type "export LILBEE_DATA=/root/data-sweep && source /root/kit/env.sh && rm -rf /root/data-sweep && lilbee"
Enter
Sleep __BOOT__s
Show
Sleep 1500ms
# Settings tour: cycle through every sub-tab via `>` and use `j` to
# scroll the longer panes (Generation, Retrieval, Crawling).
#
# Critical: do NOT Tab into individual fields. Tab focuses an Input
# widget, and once any input has focus, `>` / `q` / `/` all type as
# literal characters into the field. Stay on screen-level bindings:
# `>` (priority, cycle_pane), `j`/`k` (scroll, no focus), `q` (back).


# Two Escapes: dismiss wizard, drop chat to NORMAL.
Escape
Sleep 500ms
Escape
Sleep 500ms

# Walk to Settings: 3 `]` presses (Chat -> Catalog -> Status -> Settings).
Type "]"
Sleep 700ms
Type "]"
Sleep 700ms
Type "]"
Sleep 1500ms

# Models pane (first, default). Hold a moment.
Sleep 2s

# Ingest pane.
Type ">"
Sleep 2s

# Generation pane: 17 fields, the longest. Scroll through with `j`.
Type ">"
Sleep 2s
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms

# Retrieval pane: 20 fields. Scroll the full pane.
Type ">"
Sleep 2s

Screenshot __REEL__.png

Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms

# Display pane: short, but still scroll a bit so the panel isn't static.
Type ">"
Sleep 2s
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms

# Crawling pane (13 fields). Scroll the full pane.
Type ">"
Sleep 2s
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms

# API-Keys pane: a couple of scrolls.
Type ">"
Sleep 2s
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms

# System pane: scroll through diagnostics.
Type ">"
Sleep 2s
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms
Type "j"
Sleep 800ms

# End on the last settings pane (the payload), not back at an empty Chat screen.
