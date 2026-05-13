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

# Whichever `lilbee` is first on the shell's PATH is what the recordings show.
# Both the uv-tool / pip install and the Homebrew bundle work; pick the faster
# one in your env. (The bundled standalone binary boots in ~20s; a pip / uv
# install boots in ~2s.)
