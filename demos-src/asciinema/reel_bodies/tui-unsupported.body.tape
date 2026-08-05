Hide
Type "export LILBEE_DATA=/root/data-sweep && source /root/kit/env.sh && rm -rf /root/data-sweep && lilbee"
Enter
Sleep __BOOT__s
Show
Sleep 1500ms
# Architecture-compatibility demo: search HF Hub for "deepseek-v4" (an arch
# that isn't in the bundled llama.cpp build's MODEL_ARCH_NAMES) and
# show the rows surfacing with the `unsupported` pill in both grid
# and list view. Trying to pull one opens the ConfirmDialog gate.


# Let the app fully settle before navigating. On a pod the chat model is still
# "warming up" right after Show; if `]` fires mid-warmup it is dropped and the
# reel never leaves Chat, so `/` hits Chat's own slash binding and the query
# lands in the chat prompt as "/deepseek-v4". Generous settle fixes it (verified
# on dev721: after settle, `]` -> Catalog, `/` -> catalog search).
Sleep 3s
# Two Escapes: dismiss the setup wizard, then drop the chat input from INSERT
# to NORMAL.
Escape
Sleep 600ms
Escape
Sleep 2s

# Catalog (nav_next from Chat; priority binding). Settle before the next key.
Type "]"
Sleep 2500ms

# Chat task tab (deepseek-v4 is a chat model; Catalog defaults here, this is
# belt-and-suspenders). Numeric 1..6 select_tab are priority bindings.
Type "2"
Sleep 1500ms

# `/` focuses the catalog search input (Catalog owns slash); type + Enter fire
# the HF round-trip.
Type "/"
Sleep 600ms
Type "deepseek-v4"
Sleep 1500ms
Enter
Sleep 12s

# Grid view shows the deepseek-v4 rows, each with `unsupported` on the
# secondary pill row.
Screenshot __REEL__.png

# Switch to list view -- secondary pills become italic tags after the
# row name; "unsupported" still surfaces on every deepseek-v4 row.
Escape
Sleep 400ms
Type "v"
Sleep 2500ms
Screenshot __REEL__.png

# Back to grid so the reel closes on the stacked-pill layout.
Type "v"
Sleep 2000ms
Screenshot __REEL__.png
