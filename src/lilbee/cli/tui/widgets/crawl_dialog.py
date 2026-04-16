"""Modal dialog for configuring a web crawl."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Center, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from lilbee.cli.tui import messages as msg


@dataclass(frozen=True)
class CrawlParams:
    """Validated crawl parameters returned by CrawlDialog."""

    url: str
    depth: int
    max_pages: int


_DEFAULT_DEPTH = 0
_DEFAULT_MAX_PAGES = 50


class CrawlDialog(ModalScreen[CrawlParams | None]):
    """Modal dialog that collects URL, depth, and max-pages for a crawl."""

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
            yield Label(msg.CRAWL_DIALOG_DEPTH_LABEL, classes="crawl-field-label")
            yield Input(
                value=str(_DEFAULT_DEPTH),
                placeholder=msg.CRAWL_DIALOG_DEPTH_PLACEHOLDER,
                id="crawl-depth-input",
            )
            yield Label(msg.CRAWL_DIALOG_MAX_PAGES_LABEL, classes="crawl-field-label")
            yield Input(
                value=str(_DEFAULT_MAX_PAGES),
                placeholder=msg.CRAWL_DIALOG_MAX_PAGES_PLACEHOLDER,
                id="crawl-max-pages-input",
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
    def _parse_non_negative_int(value: str, default: int) -> int:
        """Parse a non-negative integer, returning *default* for empty strings.

        Raises ValueError on non-numeric or negative input.
        """
        if not value:
            return default
        n = int(value)
        if n < 0:
            raise ValueError("negative")
        return n

    def _try_submit(self) -> None:
        """Validate inputs and dismiss with CrawlParams or show an error."""
        from lilbee.crawler import require_valid_crawl_url

        error_widget = self.query_one("#crawl-error", Static)
        url = self.query_one("#crawl-url-input", Input).value.strip()
        depth_str = self.query_one("#crawl-depth-input", Input).value.strip()
        max_pages_str = self.query_one("#crawl-max-pages-input", Input).value.strip()

        if not url:
            error_widget.update(msg.CRAWL_DIALOG_URL_REQUIRED)
            return

        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        try:
            require_valid_crawl_url(url)
        except ValueError as exc:
            error_widget.update(msg.CRAWL_DIALOG_INVALID_URL.format(error=exc))
            return

        try:
            depth = self._parse_non_negative_int(depth_str, _DEFAULT_DEPTH)
        except ValueError:
            error_widget.update(
                msg.CRAWL_DIALOG_INVALID_NUMBER.format(field=msg.CRAWL_DIALOG_DEPTH_LABEL)
            )
            return

        try:
            max_pages = self._parse_non_negative_int(max_pages_str, _DEFAULT_MAX_PAGES)
        except ValueError:
            error_widget.update(
                msg.CRAWL_DIALOG_INVALID_NUMBER.format(field=msg.CRAWL_DIALOG_MAX_PAGES_LABEL)
            )
            return

        error_widget.update("")
        self.dismiss(CrawlParams(url=url, depth=depth, max_pages=max_pages))

    def action_cancel(self) -> None:
        self.dismiss(None)
