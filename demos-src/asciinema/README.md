# Reel pipeline

Records lilbee reels with asciinema and renders them with agg. Replaces the VHS tapes in
the parent directory for every reel that has been migrated; the rest still render from
`.tape` files through the Makefile.

```
make_reel.py <name>              # record, render, trim, gate, write the scorecard
make_reel.py <name> --no-record  # re-render and re-gate the cast already in out/
publish.py <gh-pages checkout>   # copy reels with a clean scorecard into demos/
gallery.py                       # out/reels.html: every reel beside the live asset
```

A reel is one module under `reels/`, exporting `NAME`, `COLS`, `ROWS`, `MUST_STRINGS` and
a `record(cast)` function that drives the app through `drive.Session`.

## Why this and not VHS

VHS captures pixels out of a headless Chromium. That works, but the shipped assets are
25fps of which about 91% are no-change frames padded with cursor blink, and re-recording
one meant a browser render on whatever box happened to be free. asciinema records the
byte stream with real timestamps, so a reel is a text file that renders identically
anywhere, and a failed take says which beat failed instead of producing a wrong-looking
gif.

## Things that cost takes to learn

**Render.** `agg --theme custom` is advertised by `--help` and rejected by the parser, so
the rose-pine palette goes in the cast header, and every colour needs a `#` or agg
refuses the file. The header only survives in v2, so v3 casts convert first. `--font-dir`
is additive rather than exclusive: asking for a family that is also installed system-wide
silently resolves the installed Regular, which is why the bundled face is renamed to a
family that exists nowhere else and the resolved weight is asserted. `--line-height` must
be 1.2; the 1.4 default makes the cell taller than the glyph box, and the block-drawing
characters lilbee draws panels with then cannot reach the next row, so every border
renders as dashes.

**Trim in the frame domain.** Cutting events out of a cast makes the first surviving event
paint a diff against frames that no longer exist, and the reel opens blank. Reconstructing
the screen with a terminal emulator to fix that has to reproduce every SGR attribute, and
it does not. Render the whole cast, then drop frames.

**Driving.** The pane has to be resolved from `list-panes`; `:0.0` addresses nothing when
`pane-base-index` is 1, and every capture comes back empty. Escape followed by another key
needs a delay above Textual's `ESCAPE_DELAY` or the pair arrives as `alt+<key>` and
silently does nothing. Marks are timestamps: writing a sentinel into the stream with
`send-keys -H` delivers it to the application as keystrokes, and `]` is the screen-ring
binding. Wait on content, never on the screen settling -- a Textual screen never settles.

**Navigation.** The ring is Chat, Catalog, Status, Settings, Tasks, Fleet, Sessions. Fleet
and Sessions have no palette entry; use `^g` and `^o`, and `^o` is a toggle that neither
Escape nor `q` closes. `q` is back-to-chat, not back-to-previous. Markers have to be unique
to the target screen or the walk stops early.

**Scoring.** `gates.py` writes a scorecard per reel and `UNTESTED` is not a pass. Every
threshold is asserted against a deliberately broken input by `selftest`, because a gate
that has never gone red is decoration. Frame rate is measured only inside the windows
where the driver itself was producing motion; elsewhere the cadence belongs to the model
or to a progress bar, and scoring that failed a launch reel for rendering a progress bar
correctly.
