Hide
Type "export LILBEE_DATA=/root/data-sweep && source /root/kit/env.sh && rm -rf /root/data-sweep && lilbee"
Enter
Sleep __BOOT__s
Show
Sleep 1500ms
# Command discovery surface: the Textual command palette (Ctrl+P) lists
# every registered action and filters as you type. We show three filters
# in succession so the viewer sees how the palette's a navigator, not a
# one-shot.


# Two Escapes: dismiss wizard + drop chat into NORMAL so subsequent
# Ctrl+P always reaches the app palette binding.
Escape
Sleep 500ms
Escape
Sleep 500ms

# Open palette. Default view shows every registered action.
Ctrl+P
Sleep 2s

# 1. Filter "task" -> Open Tasks (stays in rose-pine).
Type "task"
Sleep 1500ms
Enter
Sleep 1500ms

# Back to Chat so the next palette action makes sense.
Type "["
Sleep 500ms
Type "["
Sleep 500ms
Type "["
Sleep 500ms
Type "["
Sleep 800ms

# 2. Filter "model" -> Models bar focus (stays in rose-pine).
Ctrl+P
Sleep 800ms
Type "model"
Sleep 1500ms
Escape
Sleep 600ms

# 3. Slash-command catalog (stays in rose-pine).
Type "i"
Sleep 300ms
Type "/help"
Sleep 400ms
Enter
Sleep 2500ms
Escape
Sleep 600ms
# End on the slash-command catalog -- clean rose-pine (the theme-cycle step ended
# on a stale "solarized-dark" toast; drop it so the reel closes on-theme).
Screenshot __REEL__.png
