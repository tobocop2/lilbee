"""Modal dialog for configuring a web crawl."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Center, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Collapsible, Input, Label, Static

from lilbee.cli.tui import messages as msg
from lilbee.core.config import cfg


@dataclass(frozen=True)
class CrawlParams:
    """Validated crawl parameters returned by CrawlDialog.

    depth: None = whole-site unbounded. 0 = single URL only. Positive int =
    explicit link-follow depth cap. max_pages: CRAWL_PAGES_UNLIMITED (0) = no
    limit (the user cleared the field); positive int = explicit page cap.
    """

    url: str
    depth: int | None
    max_pages: int


class CrawlDialog(ModalScreen[CrawlParams | None]):
    """Modal dialog that collects URL, recursion toggle, and optional caps."""

    CSS_PATH = "crawl_dialog.tcss"
    AUTO_FOCUS = "#crawl-url-input"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(msg.CRAWL_DIALOG_TITLE, id="crawl-title")
            yield Label(msg.CRAWL_DIALOG_URL_LABEL)
            yield Input(
                placeholder=msg.CRAWL_DIALOG_URL_PLACEHOLDER,
                id="crawl-url-input",
            )
            yield Checkbox(
                msg.CRAWL_DIALOG_RECURSIVE_LABEL,
                value=True,
                id="crawl-recursive-checkbox",
            )
            # Max pages is the cap users actually reach for, so it sits at the top
            # level (not behind Advanced) prefilled with the protective default;
            # clearing it crawls unlimited without a trip to settings.
            yield Label(msg.CRAWL_DIALOG_MAX_PAGES_LABEL, classes="crawl-field-label")
            yield Input(
                value=str(cfg.crawl_safety_max_pages),
                placeholder=msg.CRAWL_DIALOG_MAX_PAGES_PLACEHOLDER,
                id="crawl-max-pages-input",
            )
            with Collapsible(title=msg.CRAWL_DIALOG_ADVANCED_TITLE, id="crawl-advanced"):
                yield Label(msg.CRAWL_DIALOG_DEPTH_LABEL, classes="crawl-field-label")
                yield Input(
                    placeholder=msg.CRAWL_DIALOG_DEPTH_PLACEHOLDER,
                    id="crawl-depth-input",
                )
            yield Static("", id="crawl-error")
            with Center():
                yield Button(msg.CRAWL_DIALOG_SUBMIT, variant="primary", id="crawl-submit")
                yield Button(msg.CRAWL_DIALOG_CANCEL, variant="default", id="crawl-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "crawl-submit":
            self._try_submit()
        else:
            self.dismiss(None)

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        self._try_submit()

    @staticmethod
    def _parse_optional_non_negative_int(value: str) -> int | None:
        """Parse a non-negative integer from *value*; empty string returns None.

        None means "no cap" in the crawl API. Zero is meaningful for the
        depth field (single-URL crawl per the crawler contract). Raises
        ValueError on non-numeric input or negative integers.
        """
        if not value:
            return None
        n = int(value)
        if n < 0:
            raise ValueError
        return n

    @staticmethod
    def _parse_max_pages(value: str) -> int:
        """Parse the max-pages field. Empty means unlimited (the user cleared it).

        Returns ``CRAWL_PAGES_UNLIMITED`` (0) for empty input, a positive int for
        an explicit cap. Raises ValueError on non-numeric or non-positive input.
        """
        from lilbee.crawler.models import CRAWL_PAGES_UNLIMITED

        if not value:
            return CRAWL_PAGES_UNLIMITED
        n = int(value)
        if n <= 0:
            raise ValueError
        return n

    def _validate(self) -> CrawlParams | str:
        """Validate inputs. Returns CrawlParams on success, error message on failure."""
        from lilbee.crawler import is_url, require_valid_crawl_url
        from lilbee.crawler.models import CRAWL_PAGES_UNLIMITED

        url = self.query_one("#crawl-url-input", Input).value.strip()
        recursive = self.query_one("#crawl-recursive-checkbox", Checkbox).value
        depth_str = self.query_one("#crawl-depth-input", Input).value.strip()
        max_pages_str = self.query_one("#crawl-max-pages-input", Input).value.strip()

        if not url:
            return msg.CRAWL_DIALOG_URL_REQUIRED

        if not is_url(url):
            url = f"https://{url}"

        try:
            require_valid_crawl_url(url)
        except ValueError as exc:
            return msg.CRAWL_DIALOG_INVALID_URL.format(error=exc)

        if not recursive:
            return CrawlParams(url=url, depth=0, max_pages=CRAWL_PAGES_UNLIMITED)

        try:
            # depth=0 means "single URL" per the crawler contract; allow it.
            depth = self._parse_optional_non_negative_int(depth_str)
        except ValueError:
            return msg.CRAWL_DIALOG_INVALID_NUMBER.format(field=msg.CRAWL_DIALOG_DEPTH_LABEL)

        try:
            max_pages = self._parse_max_pages(max_pages_str)
        except ValueError:
            return msg.CRAWL_DIALOG_INVALID_NUMBER.format(field=msg.CRAWL_DIALOG_MAX_PAGES_LABEL)

        return CrawlParams(url=url, depth=depth, max_pages=max_pages)

    def _try_submit(self) -> None:
        """Validate inputs and dismiss with CrawlParams or show an error."""
        result = self._validate()
        error_widget = self.query_one("#crawl-error", Static)
        # _validate returns str (error) or CrawlParams; isinstance disambiguates
        if isinstance(result, str):
            error_widget.update(result)
            return
        error_widget.update("")
        self.dismiss(result)

    def action_cancel(self) -> None:
        self.dismiss(None)
