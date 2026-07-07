#!/usr/bin/env bash
# Glyph specimen for the render qualification gate. Each row begins with a
# unique-hue marker block (██) so check_probes.py can locate rows by color
# even when a broken fallback font merges or reflows text bands. Rows stay
# short so they never wrap. Marker hues must match ROW_MARKERS in
# check_probes.py exactly.
row() { printf '\033[38;2;%sm██\033[0m %s' "$1" "$2"; }

row '255;0;255'   'REGULAR quick brown fox 019 O0l1'; printf '\n'
row '255;128;0'   ''; printf '\033[1mBOLD    quick brown fox 019 O0l1\033[0m\n'
row '0;128;255'   ''; printf '\033[3mITALIC quick fox\033[0m \033[1;3mBI fox\033[0m\n'
row '128;255;0'   'BOX ┌─┬┐│├┼┤╭╮╰╯┃═║▐░▒▓█'; printf '\n'
row '0;255;128'   'BRAILLE ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'; printf '\n'
row '128;0;255'   "NERD $(printf '\xee\x82\xb0 \xef\x81\xbb \xef\x84\x93 \xee\x9c\x91 \xef\x92\x89')"; printf '\n'
row '255;0;128'   'EMOJI 🐝 ✅ ⚡ 🚀'; printf '\n'
row '0;255;255'   'ANSI16 '
for i in $(seq 30 37); do printf "\033[%sm█\033[0m" "$i"; done
for i in $(seq 90 97); do printf "\033[%sm█\033[0m" "$i"; done
printf '\n'
row '255;255;0'   'TRUECOLOR '
printf '\033[38;2;235;111;146m██\033[0m'   # rose-pine love
printf '\033[38;2;246;193;119m██\033[0m'   # gold
printf '\033[38;2;156;207;216m██\033[0m'   # foam
printf '\033[38;2;0;255;0m██\033[38;2;255;0;0m██\033[38;2;0;0;255m██\033[0m\n'
row '128;128;255' "GEOMETRY $(tput cols)x$(tput lines)"; printf '\n'
