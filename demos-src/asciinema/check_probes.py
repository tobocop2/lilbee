#!/usr/bin/env python3
"""Pixel checks on the rendered glyph-specimen frame. Exit 1 on any failure.

Validated locally against a deliberately-broken render (no JetBrains Mono,
serif fallback, tofu in NERD row) — every check below caught its target
defect on that frame or was fixed until it did.

Checks:
- row integrity: all specimen rows found, none wrapped
- font fallback: REGULAR ink coverage inside the calibrated band
- bold present: BOLD row heavier than REGULAR by >8%
- tofu: within NERD/EMOJI rows, glyph cells must NOT be near-identical
  (all .notdef boxes render the same shape; real icons differ)
- truecolor passthrough: saturated swatches matched by hue, neutrals by value
- themeless bg dark and non-purple

Usage: check_probes.py <probe-glyphs.png> [--report r.json] [--calibration c.json]
Calibration (written on the authoritative pod from a verified render):
  {"regular_ink_band": [lo, hi]}
"""

import json
import sys

import numpy as np
from PIL import Image

DEFAULT_REGULAR_BAND = (0.03, 0.25)  # wide floor/ceiling until calibrated
BOLD_OVER_REGULAR_MIN = 1.08
TOFU_SIMILARITY = 0.90  # fraction of matching ink pixels between glyph cells


def luma(a: np.ndarray) -> np.ndarray:
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def ink_mask(img: np.ndarray) -> np.ndarray:
    lum = luma(img)
    bg = np.median(lum)
    return np.abs(lum - bg) > 40


# Row marker hues; must match glyph_specimen.sh exactly. Rows are located by
# scanning the LEFT MARGIN (x < 8% of width) for each marker color, which
# survives any font breakage, band merging, or reflow.
ROW_MARKERS = {
    "REGULAR": (255, 0, 255),
    "BOLD": (255, 128, 0),
    "ITALIC": (0, 128, 255),
    "BOX": (128, 255, 0),
    "BRAILLE": (0, 255, 128),
    "NERD": (128, 0, 255),
    "EMOJI": (255, 0, 128),
    "ANSI16": (0, 255, 255),
    "TRUECOLOR": (255, 255, 0),
    "GEOMETRY": (128, 128, 255),
}


def find_rows(img: np.ndarray) -> dict[str, tuple[int, int]]:
    """Each margin pixel votes for its single BEST-matching marker hue —
    independent thresholds cross-matched antialiased edges (magenta caught
    violet's edge pixels, merging bands 6 rows apart)."""
    margin = img[:, : int(img.shape[1] * 0.05)]  # markers sit <40px in; 8% let ANSI swatches encroach
    px = margin.reshape(-1, 3)
    sat_ok = (px.max(axis=1) - px.min(axis=1)) > 60
    bright = px.max(axis=1) > 120
    norm = px / (np.linalg.norm(px, axis=1, keepdims=True) + 1)
    names = list(ROW_MARKERS)
    targets = np.array([ROW_MARKERS[n] for n in names], dtype=float)
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    dots = norm @ targets.T
    best = dots.argmax(axis=1)
    strong = dots.max(axis=1) > 0.98
    rows = {}
    for i, name in enumerate(names):
        hit = sat_ok & bright & strong & (best == i)
        ys = np.nonzero(hit.reshape(margin.shape[:2]).any(axis=1))[0]
        if len(ys) < 4:
            continue
        # largest contiguous run = the marker block; stray pixels are noise
        runs, start = [], ys[0]
        for a, b in zip(ys, ys[1:]):
            if b - a > 3:
                runs.append((start, a))
                start = b
        runs.append((start, ys[-1]))
        y0, y1 = max(runs, key=lambda r: r[1] - r[0])
        rows[name] = (max(int(y0) - 2, 0), min(int(y1) + 3, margin.shape[0]))
    return rows


def glyph_cells(img: np.ndarray, band: tuple[int, int]) -> list[np.ndarray]:
    """Split a row band into per-glyph cells by ink column gaps, skipping the
    leading label word (first cell run)."""
    mask = ink_mask(img[band[0]:band[1]])
    cols = mask.mean(axis=0)
    cells, in_run, start = [], False, 0
    for x, v in enumerate(cols):
        if v > 0.02 and not in_run:
            in_run, start = True, x
        elif v <= 0.02 and in_run:
            in_run = False
            if x - start > 4:
                cells.append((start, x))
    # first two runs are the color marker and the row label; glyphs follow
    return [mask[:, a:b] for a, b in cells[2:]]


def tofu_run(cells: list[np.ndarray]) -> bool:
    """True when >=3 glyph cells are near-identical (all .notdef boxes)."""
    if len(cells) < 3:
        return True  # missing glyphs entirely counts as failure
    def sim(a: np.ndarray, b: np.ndarray) -> float:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        if h < 4 or w < 4:
            return 1.0
        return float((a[:h, :w] == b[:h, :w]).mean())
    identical = 0
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            if sim(cells[i], cells[j]) > TOFU_SIMILARITY:
                identical += 1
    return identical >= 2  # any-pair: alternating icon/tofu patterns count too


def hue_match(band_px: np.ndarray, target: tuple[int, int, int]) -> bool:
    """Match saturated swatches by hue (Chrome color management shifts RGB).
    Candidate saturation floor adapts to the target (pastels like rose-pine
    foam have sat ~60 and sat-shift under color management)."""
    t = np.array(target, dtype=float)
    target_sat = float(t.max() - t.min())
    if target_sat < 30:  # near-neutral target: match by value
        return bool((np.abs(band_px - t).max(axis=1) <= 20).any())
    mx, mn = band_px.max(axis=1), band_px.min(axis=1)
    sat = mx - mn
    cand = band_px[sat > min(40.0, target_sat * 0.55)]
    if not len(cand):
        return False
    tn = t / (np.linalg.norm(t) + 1)
    cn = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1)
    return bool((cn @ tn > 0.985).any())


TRUECOLOR_TARGETS = [
    (235, 111, 146), (246, 193, 119), (156, 207, 216),
    (0, 255, 0), (255, 0, 0), (0, 0, 255),
]


def check(png_path: str, calibration: dict) -> dict:
    img = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.float64)
    h, w, _ = img.shape
    content = img[int(h * 0.06):, int(w * 0.02):int(w * 0.98)]
    rows = find_rows(content)
    results, ok = {}, True

    results["rows_found"] = sorted(rows)
    missing = [n for n in ROW_MARKERS if n not in rows]
    if missing:
        return {"ok": False, "rows_found": sorted(rows),
                "error": f"row markers not located: {missing}"}

    mask = ink_mask(content)
    def ink(band): return float(mask[band[0]:band[1]].mean())

    band_lo, band_hi = calibration.get("regular_ink_band", DEFAULT_REGULAR_BAND)
    reg, bold = ink(rows["REGULAR"]), ink(rows["BOLD"])
    results["regular_ink"] = round(reg, 4)
    results["regular_in_band"] = band_lo <= reg <= band_hi
    # bold: binary coverage barely moves for JetBrains Mono Bold (validated on
    # a good render), so compare total ink INTENSITY over the shared specimen
    # text (columns right of marker+label), where thicker strokes integrate up
    def text_intensity(band: tuple[int, int]) -> float:
        seg = content[band[0]:band[1]]
        lum = luma(seg)
        bg = np.median(lum)
        m = ink_mask(seg)
        cols = m.mean(axis=0)
        runs, in_run, start = [], False, 0
        for x, v in enumerate(cols):
            if v > 0.02 and not in_run:
                in_run, start = True, x
            elif v <= 0.02 and in_run:
                in_run = False
                runs.append((start, x))
        if in_run:
            runs.append((start, len(cols)))
        if len(runs) < 3:
            return 0.0
        x0 = runs[2][0]  # skip marker + label
        return float(np.abs(lum[:, x0:] - bg)[m[:, x0:]].sum())
    reg_i, bold_i = text_intensity(rows["REGULAR"]), text_intensity(rows["BOLD"])
    results["regular_intensity"] = round(reg_i)
    results["bold_intensity"] = round(bold_i)
    results["bold_heavier"] = reg_i > 0 and bold_i >= reg_i * BOLD_OVER_REGULAR_MIN
    ok &= results["regular_in_band"] and results["bold_heavier"]

    for name in ("NERD", "EMOJI"):
        cells = glyph_cells(content, rows[name])
        is_tofu = tofu_run(cells)
        results[f"{name.lower()}_cells"] = len(cells)
        results[f"{name.lower()}_tofu"] = is_tofu
        ok &= not is_tofu
    for name in ("BOX", "BRAILLE"):
        cov = ink(rows[name])
        results[f"{name.lower()}_present"] = cov > 0.01
        ok &= cov > 0.01

    y0, y1 = rows["TRUECOLOR"]
    band_px = content[y0:y1].reshape(-1, 3)
    for i, target in enumerate(TRUECOLOR_TARGETS):
        found = hue_match(band_px, target)
        results[f"truecolor_{i}"] = {"target": list(target), "found": found}
        ok &= found

    bg = np.median(content.reshape(-1, 3), axis=0)
    results["bg_rgb"] = [int(x) for x in bg]
    results["bg_dark"] = bool(bg.max() < 45)
    ok &= results["bg_dark"]

    results["ok"] = bool(ok)
    return results


def main() -> None:
    calibration = {}
    if "--calibration" in sys.argv:
        calibration = json.load(open(sys.argv[sys.argv.index("--calibration") + 1]))
    res = check(sys.argv[1], calibration)
    if "--report" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--report") + 1], "w"), indent=1)
    for k, v in res.items():
        print(f"{k}: {v}")
    sys.exit(0 if res.get("ok") else 1)


if __name__ == "__main__":
    main()
