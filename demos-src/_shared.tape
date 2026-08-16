# Shared VHS settings for all lilbee demo tapes.
# Each tape `Source`s this, then sets its own `Output` and runs commands.

Set Shell "zsh"
Set Width 1400
Set Height 900
Set FontSize 18
Set Padding 30
Set TypingSpeed 35ms
Set Framerate 60
Set PlaybackSpeed 1.5
Set Theme "rose-pine-moon"
Set WindowBar Colorful

# Force truecolor so Textual renders exact rose-pine tokens. Without this, a
# 256-color host quantizes the base #191724 to cube index 16 = pure black and
# the whole theme collapses. Local shells often set it already; pods do not.
Env COLORTERM "truecolor"
Env TERM "xterm-256color"

# Whichever `lilbee` is first on the shell's PATH is what the recordings show.
# Both the uv-tool / pip install and the Homebrew bundle work; pick the faster
# one in your env. (The bundled standalone binary boots in ~20s; a pip / uv
# install boots in ~2s.)
