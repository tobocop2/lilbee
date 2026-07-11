"""The lilbee wordmark, shared by every loading surface."""

from __future__ import annotations

BEE_LINES = [
    "                                                       ",
    "@@@       @@@  @@@       @@@@@@@   @@@@@@@@  @@@@@@@@  ",
    "@@@       @@@  @@@       @@@@@@@@  @@@@@@@@  @@@@@@@@  ",
    "@@@       @@@  @@@       @@!  @@@  @@!       @@!       ",
    "@!       !@!  !@!       !@   @!@  !@!       !@!       ",
    "@!!       !!@  @!!       @!@!@!@   @!!!:!    @!!!:!    ",
    "!!!       !!!  !!!       !!!@!!!!  !!!!!:    !!!!!:    ",
    "!!:       !!:  !!:       !!:  !!!  !!:       !!:       ",
    " :!:      :!:   :!:      :!:  !:!  :!:       :!:       ",
    " :: ::::   ::   :: ::::   :: ::::   :: ::::   :: ::::  ",
    ": :: : :  :    : :: : :  :: : ::   : :: ::   : :: ::   ",
    "                                                       ",
]

LOGO_WIDTH = len(BEE_LINES[1])

# xterm-256 indexes. The logo warms from dim to bright as startup advances:
# dim while the bootstrap unpacks, pulsing while Python imports, bright once
# the engine is warm.
AMBER_DIM_XTERM = 94
AMBER_MID_XTERM = 172
AMBER_BRIGHT_XTERM = 214


def xterm_fg(index: int) -> str:
    """The ANSI escape that sets *index* as the foreground color."""
    return f"\033[38;5;{index}m"
