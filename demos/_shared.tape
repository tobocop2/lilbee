# Shared VHS settings for all lilbee demo tapes.
# Each tape `Source`s this, then sets its own `Output` and runs commands.

Set Shell "zsh"
Set Width 1400
Set Height 900
Set FontSize 18
Set Padding 30
Set TypingSpeed 55ms
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful

# Pin lilbee to the homebrew install. The recordings expect a working `lilbee`
# binary; if a venv version is broken (e.g. numpy/thinc mismatch in a pyenv
# shim), this skips that and uses the bundled standalone binary at
# /opt/homebrew/bin/lilbee.
Env PATH "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
