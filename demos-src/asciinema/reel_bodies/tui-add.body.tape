Hide
Type "export LILBEE_DATA=/root/data-add && source /root/kit/env.sh && rm -rf /root/data-add && lilbee"
Enter
Sleep __BOOT__s
Show
Sleep 1500ms
# Empty library; add a document, watch ingest in the Task Center, then ask.
# Deliberate typing speed so the command is readable; pause before submit.
Set TypingSpeed 55ms
Type "/add /root/cv-manual.pdf"
Set TypingSpeed 35ms
Sleep 2s
Enter
Sleep 1500ms
# Escape FIRST: with chat input focused, `t` types a literal. From NORMAL
# `t` opens the Task Center (priority binding).
Escape
Sleep 400ms
Type "t"
# Watch the ingest bar start filling on camera, hide the middle, come back
# for the completed row (pod index measured; remainder is hidden).
Sleep 8s
Hide
Sleep __INDEXREM__s
Show
Sleep 2500ms
# Back to chat: `[` walks Tasks -> Settings -> Status -> Catalog -> Chat.
Type "["
Sleep 400ms
Type "["
Sleep 400ms
Type "["
Sleep 400ms
Type "["
Sleep 800ms
# Confirm the corpus is live with a chat question.
Type "i"
Sleep 300ms
Type "what's my oil capacity?"
Sleep 400ms
Enter
# Show the stream starting, hide the bulk, return for the settled answer.
Sleep 4s
Hide
Sleep __GENREM__s
Show
Sleep 1200ms
Escape
Sleep 400ms
Type "G"
Sleep 2500ms
Screenshot __REEL__.png