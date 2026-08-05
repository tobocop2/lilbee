Hide
Type "export LILBEE_DATA=/root/probe-data && source /root/kit/env.sh && export OLLAMA_HOST=http://127.0.0.1:65535 && lilbee"
Enter
Sleep __BOOT__s
Show
Sleep 1500ms
# Composite tour: the chat answer streams in the background while we walk
# every screen. The async story: a local model is generating, you're free
# to keep using the UI. Original nav notes hold: `]` leaves any screen
# (priority binding), `>` cycles inner tabs, Escape drops chat to NORMAL
# without cancelling the stream.
Type "i"
Sleep 300ms
Type "what's my oil capacity?"
Sleep 500ms
Enter
Sleep 3s
Escape
Sleep 800ms
# Catalog: 6 inner tabs, `>` cycles.
Type "]"
Sleep 1800ms
Type ">"
Sleep 1500ms
Type ">"
Sleep 1500ms
Type ">"
Sleep 1500ms
Type ">"
Sleep 1500ms
Type ">"
Sleep 1500ms
# Status: open each collapsible; async workers need a beat to populate.
Type "]"
Sleep 1500ms
Enter
Sleep 1500ms
Tab
Sleep 500ms
Enter
Sleep 2500ms
Tab
Sleep 500ms
Enter
Sleep 3000ms
Tab
Sleep 500ms
Enter
Sleep 2000ms
# Settings: 8 sub-tabs.
Type "]"
Sleep 1500ms
Type ">"
Sleep 1200ms
Type ">"
Sleep 1200ms
Type ">"
Sleep 1200ms
Type ">"
Sleep 1200ms
Type ">"
Sleep 1200ms
Type ">"
Sleep 1200ms
Type ">"
Sleep 1200ms
# Tasks.
Type "]"
Sleep 2500ms
# Help cheat sheet.
Type "?"
Sleep 2500ms
Escape
Sleep 500ms
# Back to Chat; slash catalog glance.
Type "]"
Sleep 1500ms
Type "i"
Sleep 300ms
Type "/help"
Sleep 400ms
Enter
Sleep 2500ms
Escape
Sleep 500ms
# Let the answer finish if the walk beat it (measured remainder), then read it.
Hide
Sleep __GENREM__s
Show
Sleep 1000ms
Escape
Sleep 400ms
Type "G"
Sleep 2500ms
Screenshot __REEL__.png