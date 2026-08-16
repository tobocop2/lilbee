#!/usr/bin/env python3
"""Build the local browser review page for a batch of reel artifacts.

Reads a gathered reels-out tree (mp4 + png + qa json + FAILED.txt per reel),
generates display-size gifs via gif_pipeline.sh, and writes review.html with
mp4 players, gifs at true display size, poster frames, autoqa verdicts, and
probe frames. Open with `open review.html`.

Usage: build_review.py reels.yaml <reels-out-dir> <out-dir> [--mode lanczos] [--dither bayer:bayer_scale=4]
"""

import html
import json
import pathlib
import subprocess
import sys

import yaml

HERE = pathlib.Path(__file__).parent


def gif_for(reel: str, mp4: pathlib.Path, display: tuple[int, int],
            out_dir: pathlib.Path, mode: str, dither: str) -> pathlib.Path | None:
    gif = out_dir / f"{reel}.gif"
    r = subprocess.run(
        ["bash", str(HERE / "gif_pipeline.sh"), str(mp4), str(gif),
         str(display[0]), str(display[1]), mode, dither],
        capture_output=True, text=True,
    )
    print(r.stdout.strip())
    return gif if gif.exists() else None


def main() -> None:
    manifest = yaml.safe_load(open(sys.argv[1]))
    src = pathlib.Path(sys.argv[2])
    out = pathlib.Path(sys.argv[3])
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "lanczos"
    dither = sys.argv[sys.argv.index("--dither") + 1] if "--dither" in sys.argv else "bayer:bayer_scale=4"
    out.mkdir(parents=True, exist_ok=True)

    cards = []
    for reel_dir in sorted(p for p in src.iterdir() if p.is_dir() and not p.name.startswith("_")):
        reel = reel_dir.name
        r = manifest["reels"].get(reel, {})
        cls = manifest["classes"].get(r.get("class", ""), {})
        display = (cls.get("width", 1400), cls.get("height", 900))
        mp4 = reel_dir / f"{reel}.mp4"
        qa_path = reel_dir / f"qa-{reel}.json"
        failed = reel_dir / "FAILED.txt"
        qa = json.loads(qa_path.read_text()) if qa_path.exists() else {}

        body = []
        status = "FAILED" if failed.exists() else ("PASS" if qa.get("ok") else ("QA-FAIL" if qa else "QA-UNKNOWN"))
        if mp4.exists():
            rel_mp4 = out / mp4.name
            rel_mp4.write_bytes(mp4.read_bytes())
            body.append(f'<video controls preload="metadata" style="width:100%" src="{mp4.name}"></video>')
            gif = gif_for(reel, mp4, display, out, mode, dither)
            if gif:
                body.append(f'<h4>gif at display size {display[0]}x{display[1]}</h4>'
                            f'<img src="{gif.name}" style="max-width:100%">')
        png = reel_dir / f"{reel}.png"
        if png.exists():
            (out / png.name).write_bytes(png.read_bytes())
            body.append(f'<h4>poster</h4><img src="{png.name}" style="max-width:100%">')
        if failed.exists():
            body.append(f"<pre style='color:#c33'>{html.escape(failed.read_text()[:4000])}</pre>")
        if qa:
            body.append(f"<details><summary>autoqa report</summary><pre>{html.escape(json.dumps(qa, indent=1))}</pre></details>")
        color = {"PASS": "#2a2", "FAILED": "#c33", "QA-FAIL": "#c33"}.get(status, "#a80")
        cards.append(
            f'<section style="margin:2em 0;border:1px solid #444;border-radius:8px;padding:1em">'
            f'<h2>{reel} <span style="color:{color}">[{status}]</span></h2>{"".join(body)}</section>'
        )

    # probe frames if present
    for probe in sorted(src.glob("_qa/*.png")) + sorted(src.glob("authoritative-probe.png")):
        (out / probe.name).write_bytes(probe.read_bytes())
        cards.insert(0, f'<section><h2>render qualification probe: {probe.name}</h2>'
                        f'<img src="{probe.name}" style="max-width:100%"></section>')

    (out / "review.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>reel review</title>"
        "<body style='background:#191724;color:#e0def4;font-family:system-ui;max-width:1500px;margin:2em auto;padding:0 1em'>"
        f"<h1>reel batch review — {len(cards)} items</h1>" + "".join(cards)
    )
    print(f"open {out}/review.html")


if __name__ == "__main__":
    main()
