Hide
# Fresh empty library; Qwen3-4B chat model (matches the original, loads fast so
# the on-camera answer renders promptly even cold).
Type "export LILBEE_DATA=/root/data-whatis && source /root/kit/env.sh && export LILBEE_CHAT_MODEL='Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf' && rm -rf /root/data-whatis && lilbee"
Enter
Sleep __BOOT__s
Show
Sleep 1500ms
# Full flow, faithful to the original: add lilbee's own README on camera (the
# file-autocomplete dropdown surfaces the repo's sibling files), watch it index
# in the Task Center, then ask what lilbee is and get a cited answer.
# Deliberate typing so the /add command + autocomplete are readable.
Set TypingSpeed 55ms
Type "/add ~/lilbee-repo/README.md"
Set TypingSpeed 35ms
Sleep 2s
Enter
Sleep 1500ms
# Escape to NORMAL, `t` opens the Task Center; the add task fills to 100% done.
Escape
Sleep 400ms
Type "t"
Sleep 2s
Hide
Sleep __INDEXREM__s
Show
Sleep 1500ms
# Back to Chat: `[` walks Tasks -> Settings -> Status -> Catalog -> Chat.
Type "["
Sleep 400ms
Type "["
Sleep 400ms
Type "["
Sleep 400ms
Type "["
Sleep 800ms
# Ask the one-sentence question; cited answer from the README.
Type "i"
Sleep 400ms
Type@15ms "what is lilbee in one sentence?"
Sleep 1000ms
Enter
Sleep __GEN__s
Sleep 2500ms
Screenshot __REEL__.png