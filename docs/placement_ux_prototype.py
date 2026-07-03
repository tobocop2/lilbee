"""Placement screen UX prototype — calm overview + per-model drill-in + advanced controls.

A design reference for the real TUI placement screen, not production code.
Grounded in lilbee's real fleet: 4 models (Chat, Embeddings, Reranker, Vision).
Chat/Reranker are single instances (tensor-split a big model across cards);
Embeddings/Vision are data-parallel (a copy per card) for throughput. LM-Studio
style: an overview, then one model at a time. Fake data, no fleet.

The drill-in has an Advanced section (press 'a') that exposes the full
RolePlacement surface: distribute mode (copies vs split) for replicable roles,
custom per-card tensor-split weights, and removing an optional model.

NOTE for the real implementation: this mock is a single Static with manual
on_key handling, so it is NOT tab-navigable. The production screen must use
real focusable widgets (one per GPU checkbox / advanced control) so Tab and
arrow keys both work, consistent with the rest of the TUI.
Run: uv run --with textual python proto.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

from textual.app import App, ComposeResult
from textual.widgets import Static

# rose-pine
BG = "#191724"
TEXT = "#e0def4"
SUB = "#908caa"
MUTE = "#6e6a86"
IRIS = "#c4a7e7"
FOAM = "#9ccfd8"
GOLD = "#f6c177"
PINE = "#3e8fb0"
ROSE = "#ebbcba"
LOVE = "#eb6f92"
TRACK = "#26233a"
SEL = "#403d52"
GPU_NAME = "RTX 4090"
CARD_GB = 24.0


@dataclass
class Role:
    key: str
    label: str
    model: str
    job: str
    size_gb: float
    color: str
    split: bool  # True = tensor-split one model across cards; False = a copy per card
    required: bool = True  # required roles (chat/embed) can't be removed
    replicable: bool = False  # embed/vision can choose copies vs split
    enabled: bool = True
    devices: set[int] = field(default_factory=set)
    weights: dict[int, int] | None = None  # custom tensor-split ratio per device; None = even


class PlacementProto(App):
    CSS = f"""
    Screen {{ background: {BG}; color: {TEXT}; }}
    #screen {{ padding: 1 4; }}
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self.gpu_count = 4
        self.mode = "overview"  # overview | detail
        self.ov = 0  # focused model row in overview
        self.det = 0  # focused item index in detail
        self.adv = False  # advanced section expanded in detail
        self.detail_key = ""
        self.toast = ""
        self.roles: list[Role] = [
            Role(
                "chat",
                "Chat",
                "Qwen2.5 72B",
                "the model you talk to",
                56.0,
                FOAM,
                split=True,
                required=True,
            ),
            Role(
                "embed",
                "Embeddings",
                "nomic-embed",
                "indexes docs for search",
                0.3,
                GOLD,
                split=False,
                required=True,
                replicable=True,
            ),
            Role(
                "rerank",
                "Reranker",
                "bge-reranker",
                "sharpens search results",
                0.6,
                PINE,
                split=True,
                required=False,
                enabled=False,
            ),
            Role(
                "vision",
                "Vision",
                "Qwen2-VL OCR",
                "reads images & PDFs",
                5.0,
                ROSE,
                split=False,
                required=False,
                replicable=True,
                enabled=False,
            ),
        ]
        self.auto_layout()

    # -- model -----------------------------------------------------------
    def gpus(self) -> list[int]:
        return list(range(self.gpu_count))

    def auto_layout(self) -> None:
        for r in self.roles:
            if not r.enabled:
                r.devices = set()
                continue
            need = _cards_needed(r.size_gb, self.gpu_count) if r.split else 1
            r.devices = set(range(min(max(1, need), self.gpu_count)))

    def is_auto(self) -> bool:
        for r in self.roles:
            need = _cards_needed(r.size_gb, self.gpu_count) if r.split else 1
            want = set(range(min(max(1, need), self.gpu_count))) if r.enabled else set()
            if r.devices != want:
                return False
        return True

    def chunk(self, r: Role, gpu: int) -> float:
        if not r.enabled or gpu not in r.devices or not r.devices:
            return 0.0
        if not r.split:
            return r.size_gb  # a copy on each card
        if r.weights:  # custom tensor-split ratio
            tot = sum(r.weights.get(g, 1) for g in r.devices) or 1
            return r.size_gb * r.weights.get(gpu, 1) / tot
        return r.size_gb / len(r.devices)  # even split

    def _detail_items(self, r: Role) -> list[str]:
        items = [f"gpu:{g}" for g in self.gpus()]
        if self.adv:
            if r.replicable:
                items.append("mode")
            if r.split and len(r.devices) > 1:
                items.append("weights")
                if r.weights is not None:
                    items += [f"w:{g}" for g in sorted(r.devices)]
            if not r.required:
                items.append("remove")
        return items

    def used(self, gpu: int) -> float:
        return sum(self.chunk(rr, gpu) for rr in self.roles)

    def _role(self, key: str) -> Role:
        return next(r for r in self.roles if r.key == key)

    def _runs_on(self, r: Role) -> str:
        """Colored 'runs on' summary padded to a fixed VISIBLE width (tags excluded)."""
        if not r.enabled:
            body, plain = f"[{MUTE}]—[/]", "—"
        elif not r.devices:
            body, plain = f"[{LOVE}]no GPU[/]", "no GPU"
        else:
            d = sorted(r.devices)
            gpus = "GPU " + "·".join(map(str, d))
            if r.split and len(d) > 1:
                note = f"split across {len(d)}"
            elif len(d) > 1:
                note = "a copy on each"
            else:
                note = ""
            plain = (gpus + ("  " + note if note else "")).rstrip()
            body = f"[{TEXT}]{gpus}[/]" + (f"  [{MUTE}]{note}[/]" if note else "")
        return body + " " * max(0, 30 - len(plain))

    def _role_ok(self, r: Role) -> tuple[bool, str]:
        if not r.enabled:
            return True, ""
        if not r.devices:
            return False, f"[{LOVE}]add a GPU[/]"
        each = self.chunk(r, sorted(r.devices)[0])
        if each > CARD_GB:
            return False, f"[{LOVE}]too big — add a card[/]"
        if any(self.used(g) > CARD_GB for g in r.devices):
            return False, f"[{GOLD}]a card is full[/]"
        return True, f"[{FOAM}]✓[/]"

    def _bar(self, gpu: int, width: int) -> str:
        used = self.used(gpu)
        over = used > CARD_GB
        denom = used if over else CARD_GB
        out, wsum = "", 0
        for rr in self.roles:
            c = self.chunk(rr, gpu)
            if c > 0:
                w = max(1, round(c / denom * width))
                out += f"[{rr.color}]{'█' * w}[/]"
                wsum += w
        if not over:
            out += f"[{TRACK}]{'░' * max(0, width - wsum)}[/]"
        return out

    # -- overview --------------------------------------------------------
    def _overview(self) -> str:
        L: list[str] = []
        L.append(f"[{MUTE}]Chat · Catalog · Status · Settings · Tasks · [/][{IRIS}]Placement[/]")
        L.append("")
        status = (
            f"[{FOAM}]● Automatic[/] [{MUTE}](lilbee chose this)[/]"
            if self.is_auto()
            else f"[{GOLD}]● Custom[/] [{MUTE}](^r reset to automatic)[/]"
        )
        L.append(f"[{TEXT}]Placement[/]      {status}")
        L.append("")
        L.append(f"[{SUB}]Your models[/]                                 [{SUB}]Runs on[/]")
        for i, r in enumerate(self.roles):
            focus = i == self.ov
            arrow = f"[{IRIS}]▸[/]" if focus else " "
            name = f"[{r.color}]{r.label:<11}[/][{MUTE}]{r.model:<14}[/]"
            ok, tag = self._role_ok(r)
            trailing = (f"[{SUB} on {SEL}] enable [/]" if focus else "") if not r.enabled else tag
            line = f"  {arrow} {name}  {self._runs_on(r)} {trailing}"
            if focus:
                line = f"[on {SEL}]{line}[/]"
            L.append(line)
        L.append("")
        L.append(f"[{SUB}]Your GPUs[/]")
        cells = []
        for g in self.gpus():
            used = self.used(g)
            tag = (
                f"[{LOVE}]{used:.0f}/{CARD_GB:.0f} over[/]"
                if used > CARD_GB
                else (f"[{SUB}]{used:.0f}/{CARD_GB:.0f}[/]" if used > 0 else f"[{MUTE}]free[/]")
            )
            cells.append(f"[{TEXT}]GPU {g}[/] {self._bar(g, 8)} {tag}")
        # wrap 4 per line
        for i in range(0, len(cells), 4):
            L.append("  " + "    ".join(cells[i : i + 4]))
        L.append("")
        all_ok = all(self._role_ok(r)[0] for r in self.roles)
        fit = (
            f"[{FOAM}]Everything fits ✓[/]"
            if all_ok
            else f"[{LOVE}]Some models don't fit — open one to fix.[/]"
        )
        L.append(fit)
        L.append("")
        L.append(
            f"[{MUTE}]↑↓ model · enter customize · ^a auto · ^r reset · ^s apply · "
            f"1/2/4/8 GPUs · q[/]"
        )
        if self.toast:
            L.append(f"\n[{FOAM}]{self.toast}[/]")
        return "\n".join(L)

    # -- detail ----------------------------------------------------------
    def _detail(self) -> str:
        r = self._role(self.detail_key)
        items = self._detail_items(r)
        self.det = max(0, min(len(items) - 1, self.det))
        cur = items[self.det] if items else ""
        L: list[str] = []
        size_note = (
            "[{}]too big for one card[/]".format(LOVE)
            if r.size_gb > CARD_GB
            else f"[{MUTE}]{r.size_gb:.0f} GB · {r.job}[/]"
        )
        L.append(
            f"[{MUTE}]‹ back[/]     [{r.color}]{r.label}[/] [{MUTE}]— {r.model}[/]   {size_note}"
        )
        L.append("")
        L.append(f"[{TEXT}]Which GPUs?[/]")
        for g in self.gpus():
            on = g in r.devices
            box = f"[{r.color}]☑[/]" if on else f"[{MUTE}]☐[/]"
            used = self.used(g)
            tag = (
                f"[{LOVE}]{used:.0f}/{CARD_GB:.0f} full[/]"
                if used > CARD_GB
                else (f"[{SUB}]{used:.0f}/{CARD_GB:.0f} GB[/]" if used > 0 else f"[{MUTE}]free[/]")
            )
            focus = cur == f"gpu:{g}"
            line = (
                f"   {box}  [{TEXT}]GPU {g}[/]  [{MUTE}]{GPU_NAME}[/]   {self._bar(g, 10)}  {tag}"
            )
            L.append(f"[{IRIS}]▸[/]{line[1:]}" if focus else f" {line}")
        L.append("")
        # how + fit
        d = sorted(r.devices)
        each = self.chunk(r, d[0]) if d else 0
        if r.split:
            how = (
                f"[{TEXT}]How:[/]  one model [{r.color}]split[/] across the selected cards"
                if len(d) > 1
                else f"[{TEXT}]How:[/]  runs on the selected card"
            )
        else:
            how = (
                f"[{TEXT}]How:[/]  [{r.color}]a copy on each[/] selected card [{MUTE}](more = faster {r.job.split()[0]})[/]"
                if len(d) > 1
                else f"[{TEXT}]How:[/]  one copy on the selected card"
            )
        if not d:
            fit = f"[{LOVE}]Pick at least one GPU.[/]"
        elif each > CARD_GB:
            fit = f"[{LOVE}]Still too big — select another card.[/]"
        elif any(self.used(g) > CARD_GB for g in d):
            fit = f"[{GOLD}]Fits, but a card is over budget.[/]"
        else:
            spread = self._split_spread(r, d) if (r.split and len(d) > 1) else ""
            detail = spread or (
                f"{each:.0f} GB on each of {len(d)} cards"
                if len(d) > 1
                else f"{each:.0f} GB on Card {d[0]}"
            )
            fit = f"[{FOAM}]✓ Fits[/] [{MUTE}]— {detail}[/]"
        L.append(how)
        L.append(f"      {fit}")
        L.append("")
        L += self._advanced_block(r, cur)
        L.append("")
        L.append(self._detail_hints(r, cur))
        return "\n".join(L)

    def _split_spread(self, r: Role, d: list[int]) -> str:
        return "split " + " / ".join(f"{self.chunk(r, g):.0f}" for g in d) + " GB"

    def _advanced_block(self, r: Role, cur: str) -> list[str]:
        controls = []
        if r.replicable:
            controls.append("distribute mode")
        if r.split and len(r.devices) > 1:
            controls.append("custom split weights")
        if not r.required:
            controls.append("remove model")
        if not controls:
            return [f"[{MUTE}]No advanced options for this model.[/]"]
        arrow = "▾" if self.adv else "▸"
        hint = "" if self.adv else f"  [{MUTE}]({', '.join(controls)})[/]"
        out = [f"[{MUTE}]{arrow}[/] [{SUB}]Advanced[/]{hint}"]
        if self.adv:
            out += self._advanced_lines(r, cur)
        return out

    def _advanced_lines(self, r: Role, cur: str) -> list[str]:
        out: list[str] = []

        def mark(item: str) -> str:
            return f"[{IRIS}]▸[/]" if cur == item else " "

        def radio(on: bool, label: str) -> str:
            return f"[{r.color}]● {label}[/]" if on else f"[{MUTE}]○ {label}[/]"

        if r.replicable:
            choice = f"{radio(not r.split, 'Copies')}   {radio(r.split, 'Split')}"
            out.append(f"  {mark('mode')} [{TEXT}]Distribute[/]   {choice}")
            if cur == "mode":
                out.append(
                    f"      [{MUTE}]Copies = a full model per card (faster). Split = one model across cards (for big models).[/]"
                )
        if r.split and len(r.devices) > 1:
            choice = (
                f"{radio(r.weights is None, 'Even')}   {radio(r.weights is not None, 'Custom')}"
            )
            out.append(f"  {mark('weights')} [{TEXT}]Split weights[/]  {choice}")
            if r.weights is not None:
                for g in sorted(r.devices):
                    w = r.weights.get(g, 1)
                    bar = f"[{r.color}]{'▮' * w}{'·' * (9 - w)}[/]"
                    tip = f"  [{MUTE}](+/- to adjust)[/]" if cur == f"w:{g}" else ""
                    out.append(
                        f"  {mark(f'w:{g}')}   [{MUTE}]GPU {g}[/]  {bar} [{MUTE}]weight {w} → {self.chunk(r, g):.0f} GB[/]{tip}"
                    )
        if not r.required:
            tip = (
                f"  [{MUTE}](frees its VRAM; re-add it from the overview)[/]"
                if cur == "remove"
                else ""
            )
            out.append(f"  {mark('remove')} [{LOVE}]Remove this model[/]{tip}")
        return out

    def _detail_hints(self, r: Role, cur: str) -> str:
        nav = "↑↓ move · +/- adjust" if cur.startswith("w:") else "↑↓ move · space toggle"
        has_adv = r.replicable or (r.split and len(r.devices) > 1) or not r.required
        adv = "a advanced · " if has_adv else ""
        return f"[{MUTE}]{nav} · {adv}enter done · esc back[/]"

    # -- lifecycle -------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Static(id="screen")

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        body = self._overview() if self.mode == "overview" else self._detail()
        self.query_one("#screen", Static).update(body)

    def on_key(self, event) -> None:  # noqa: C901
        k = event.key
        self.toast = ""
        if k in ("1", "2", "4", "8"):
            self.gpu_count = int(k)
            self.ov = self.det = 0
            self.auto_layout()
            return self._refresh()
        if self.mode == "overview":
            if k in ("down", "j"):
                self.ov = min(len(self.roles) - 1, self.ov + 1)
            elif k in ("up", "k"):
                self.ov = max(0, self.ov - 1)
            elif k == "enter":
                r = self.roles[self.ov]
                if not r.enabled:
                    r.enabled = True
                    self.auto_layout()
                else:
                    self.detail_key = r.key
                    self.det = 0
                    self.adv = False
                    self.mode = "detail"
            elif k == "ctrl+a":
                self.auto_layout()
                self.toast = "Arranged automatically."
            elif k == "ctrl+r":
                self.auto_layout()
            elif k == "ctrl+s":
                self.toast = "Applied — fleet reconfigured."
        else:  # detail
            r = self._role(self.detail_key)
            items = self._detail_items(r)
            self.det = max(0, min(len(items) - 1, self.det))
            cur = items[self.det] if items else ""
            if k in ("down", "j"):
                self.det = min(len(items) - 1, self.det + 1)
            elif k in ("up", "k"):
                self.det = max(0, self.det - 1)
            elif k == "a":
                self.adv = not self.adv
            elif k in ("+", "=") and cur.startswith("w:") and r.weights is not None:
                g = int(cur[2:])
                r.weights[g] = min(9, r.weights.get(g, 1) + 1)
            elif k in ("-", "_") and cur.startswith("w:") and r.weights is not None:
                g = int(cur[2:])
                r.weights[g] = max(1, r.weights.get(g, 1) - 1)
            elif k == "space" or (k == "enter" and cur in ("mode", "weights", "remove")):
                self._detail_act(r, cur)
            elif k == "escape":
                self.mode = "overview"
            elif k == "enter":
                self.mode = "overview"
        self._refresh()

    def _detail_act(self, r: Role, cur: str) -> None:
        if cur.startswith("gpu:"):
            g = int(cur[4:])
            if g in r.devices:
                if len(r.devices) > 1:
                    r.devices.discard(g)
            else:
                r.devices.add(g)
        elif cur == "mode":
            r.split = not r.split
            r.weights = None
        elif cur == "weights":
            r.weights = None if r.weights is not None else {g: 1 for g in sorted(r.devices)}
        elif cur.startswith("w:") and r.weights is not None:
            g = int(cur[2:])
            r.weights[g] = (r.weights.get(g, 1) % 9) + 1
        elif cur == "remove":
            r.enabled = False
            r.devices.clear()
            r.weights = None
            self.adv = False
            self.mode = "overview"


def _cards_needed(size_gb: float, available: int) -> int:
    per = CARD_GB * 0.9
    n = 1
    while size_gb / n > per and n < max(1, available):
        n += 1
    return n


if __name__ == "__main__":
    PlacementProto().run()
