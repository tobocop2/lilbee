# 6. Reasoning tag parsing and stream position tracking

## Status
Accepted

## Context
When `show_reasoning=False`, reasoning tokens from models like smollm2 have empty content. A model that opens `<think>` and never closes it caused `filter_reasoning` to loop forever because the reasoning character cap never triggered.

## Finding
Character counting based on visible output fails for hidden content. The reasoning cap must track raw chars consumed from the stream regardless of the `show_reasoning` flag.

## Decision
Move reasoning char tracking into `_TagParser` to count raw chars consumed. Check the cap after each token, not inside the yield loop. Added a drain cap to prevent hangs on stuck reasoning loops.

## Consequences
- Models with unclosed reasoning tags no longer hang streaming or the TUI
- Cap enforcement works correctly even when reasoning output is suppressed
- General principle: counting/limiting must track the raw stream, not filtered output
