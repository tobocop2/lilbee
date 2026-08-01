#!/usr/bin/env python3
"""Render every finished reel to one browsable page: gif, scorecard, and the A/B.

Fills in as reels land. Anything in out/ with a gif is listed; a reel with a failing or
untested scorecard row is shown as NOT DONE with the row that blocked it, so the page is
a production board rather than a highlight reel.
"""
from __future__ import annotations

import html
import pathlib

KIT = pathlib.Path(__file__).resolve().parent
OUT = KIT / "out"
PAGE = OUT / "reels.html"
GHP = pathlib.Path("/tmp/ghp/demos")

CSS = """
:root{--ink:#e0def4;--bg:#191724;--panel:#1f1d2e;--line:#2a2837;--muted:#908caa;
      --foam:#9ccfd8;--love:#eb6f92;--gold:#f6c177;--iris:#c4a7e7;
      --mono:ui-monospace,"SF Mono",Menlo,monospace;
      --sans:-apple-system,"Helvetica Neue",Arial,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px}
.wrap{max-width:1180px;margin:0 auto;padding:44px 28px 80px}
h1{font-size:30px;margin:0 0 4px;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:13.5px;margin:0 0 34px}
.reel{border:1px solid var(--line);border-radius:10px;background:var(--panel);
      padding:20px 22px;margin:0 0 22px}
.hd{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.nm{font-family:var(--mono);font-size:17px;font-weight:600}
.tag{font-size:10.5px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;
     padding:3px 9px;border-radius:20px}
.ok{background:var(--foam);color:#10202a}.no{background:var(--gold);color:#2a1d00}
img{max-width:100%;border-radius:6px;display:block;border:1px solid var(--line)}
.cols{display:grid;grid-template-columns:1fr;gap:16px}
@media(min-width:900px){.cols.ab{grid-template-columns:1fr 1fr}}
.cap{color:var(--muted);font-size:12px;margin:6px 0 0;font-family:var(--mono)}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:14px}
td{padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
td:first-child{font-family:var(--mono);white-space:nowrap;width:1%}
.p{color:var(--foam)}.f{color:var(--love)}.u{color:var(--gold)}
.none{color:var(--muted);padding:40px 0;text-align:center}
"""


def scorecard_rows(name: str) -> list[tuple[str, str, str]]:
    """Read the scorecard the runner wrote, rather than re-deriving one here.

    Re-running the gates from this page would score with different must-strings and a
    different trim window than the run that produced the gif, so the page could show a
    reel as passing checks it never passed.
    """
    score = OUT / f"{name}.score.txt"
    if not score.exists():
        return []
    rows = []
    for line in score.read_text().splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        mark = line[1:5].strip()
        rest = line[6:].split(":", 1)
        if len(rest) == 2:
            rows.append((mark, rest[0].strip(), rest[1].strip()))
    return rows


def build() -> pathlib.Path:
    # live-*.gif are copies of the shipped assets used for the side-by-side, not reels.
    gifs = sorted(p for p in OUT.glob("*.gif")
                  if not p.stem.endswith("-contact") and not p.stem.startswith("live-"))
    cards = []
    for gif in gifs:
        name = gif.stem
        rows = scorecard_rows(name)
        bad = [r for r in rows if r[0] in ("FAIL", "----")]
        tag = ("no", "NOT DONE") if bad or not rows else ("ok", "READY")
        blocked = f" &mdash; blocked by {html.escape(bad[0][1])}" if bad else ""
        body = [f'<div class="reel"><div class="hd"><span class="nm">{html.escape(name)}</span>'
                f'<span class="tag {tag[0]}">{tag[1]}</span>'
                f'<span class="cap">{blocked}</span></div>']
        # Copy the live asset next to the page: a file:// reference from an http-less
        # page is blocked by the browser, so the comparison silently renders empty.
        live_src = GHP / f"{name}.gif"
        live = OUT / f"live-{name}.gif"
        if live_src.exists() and not live.exists():
            import shutil; shutil.copy(live_src, live)
        if live.exists():
            body.append('<div class="cols ab">'
                        f'<div><img src="{gif.name}"><p class="cap">new pipeline (asciinema + agg)</p></div>'
                        f'<div><img src="{live.name}"><p class="cap">live on gh-pages (VHS)</p></div>'
                        '</div>')
        else:
            body.append(f'<div class="cols"><div><img src="{gif.name}">'
                        '<p class="cap">new pipeline (no live asset to compare)</p></div></div>')
        if rows:
            trs = "".join(
                f'<tr><td class="{ {"PASS":"p","FAIL":"f","----":"u"}[m] }">{m}</td>'
                f"<td>{html.escape(n)}</td><td>{html.escape(d)}</td></tr>"
                for m, n, d in rows)
            body.append(f"<table>{trs}</table>")
        body.append("</div>")
        cards.append("".join(body))

    inner = "".join(cards) or '<p class="none">No finished reels yet.</p>'
    PAGE.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>lilbee reels</title>"
        f"<style>{CSS}</style></head><body><div class='wrap'>"
        "<h1>lilbee demo reels</h1>"
        f"<p class='sub'>{len(gifs)} rendered &middot; each shown against its live "
        "counterpart where one exists, with the scorecard that decides whether it ships"
        "</p>" + inner + "</div></body></html>")
    return PAGE


if __name__ == "__main__":
    print(build())
