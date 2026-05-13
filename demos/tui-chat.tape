Source demos/_shared.tape
Output demos/_out/tui-chat.gif

# Chat with the Crown Victoria owner's manual. Models + index are pre-staged
# by demos/_prep.sh; LILBEE_DATA points at the prepared corpus.
#
# Measured TUI timings:
#   lilbee TUI cold launch -> ready             1.02s -> Sleep 3s
#   First chat (cold model load + answer)         ~14s -> Sleep 18s
#   Warm follow-up (answer + cite + idle settle)  ~10s -> Sleep 12s
#
# Each `Type "..."` is followed by `Sleep 500ms` then `Enter`; Textual's
# Input widget can miss the Submit if the keys arrive in the same frame as
# the last character of input.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-chat"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 500ms
Enter
Sleep 3s

# Signature question (matches the README copy). Cold model load is folded
# into this first answer.
Type "what does the oil pressure warning mean?"
Sleep 500ms
Enter
Sleep 18s

Screenshot demos/_out/tui-chat.png

# Single-fact follow-ups Qwen3 0.6B nails cleanly from the corpus.
Type "what is my oil capacity?"
Sleep 500ms
Enter
Sleep 12s

Type "what tire pressure does it recommend?"
Sleep 500ms
Enter
Sleep 12s

Type "what fuel does it take?"
Sleep 500ms
Enter
Sleep 12s

Type "/quit"
Sleep 500ms
Enter
Sleep 1s
