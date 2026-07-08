Set Shell "bash"
Set Width 1400
Set Height 900
Set FontSize 14
Set Padding 20
Set TypingSpeed 35ms
Set Framerate 25
Set PlaybackSpeed 1
Set Theme "rose-pine-moon"
Set WindowBar Colorful
Env COLORTERM "truecolor"
Env TERM "xterm-256color"
Output agent-hermes.mp4
Output agent-hermes.gif
Hide
Type "source /workspace/kit/k5_env.sh && cd /workspace/lilbee && lilbee launch hermes"
Enter
Sleep 90s
Show
Sleep 3s
Type "Use the lilbee search tool to find where chat tensor splits are computed, and explain in two sentences which files decide the split."
Sleep 1200ms
Enter
Sleep 75s
Sleep 3s
Screenshot agent-hermes.png
