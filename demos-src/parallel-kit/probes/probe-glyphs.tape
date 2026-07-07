Set Shell "bash"
Set Width 2800
Set Height 1800
Set FontSize 28
Set FontFamily "JetBrains Mono, Symbols Nerd Font Mono, Noto Color Emoji"
Set LetterSpacing 2
Set Padding 40
Set Framerate 25
Set PlaybackSpeed 1
Set WindowBar Colorful
Set WindowBarSize 60
Env COLORTERM "truecolor"
Env TERM "xterm-256color"
Output probe-glyphs.mp4

Hide
Type "clear && bash /workspace/kit/glyph_specimen.sh"
Enter
Sleep 3s
Show
Sleep 4s
Screenshot probe-glyphs.png
Sleep 1s
