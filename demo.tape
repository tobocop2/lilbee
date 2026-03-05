# lilbee demo — commands overview + Crown Victoria Q&A
Output demo.gif
Set Theme "Catppuccin Mocha"
Set FontSize 14
Set Width 1200
Set Height 800
Set Padding 10
Set TypingSpeed 20ms
Set Shell "bash"

Require lilbee

Hide
Type "export COLUMNS=120"
Enter
Sleep 500ms
Show

# --- Status: show what's indexed ---
Type "lilbee status"
Enter
Sleep 4s

# --- Enter chat mode ---
Type "lilbee chat"
Enter
Sleep 2s

# --- /help: show available commands ---
Type "/help"
Enter
Sleep 2s

# --- /status: check index from inside chat ---
Type "/status"
Enter
Sleep 4s

# --- Ask questions about the Crown Victoria ---
Type "What size engine does my Crown Victoria have?"
Enter
Wait+Screen@60s /Sources:/
Sleep 2s

Type "How do I change a flat tire?"
Enter
Wait+Screen@60s /Sources:/
Sleep 2s

Type "How often should I change the oil?"
Enter
Wait+Screen@60s /Sources:/
Sleep 2s

Type "What is the part number for my headlamps?"
Enter
Wait+Screen@60s /Sources:/
Sleep 2s

# --- Exit ---
Type "/quit"
Enter
Sleep 1s
