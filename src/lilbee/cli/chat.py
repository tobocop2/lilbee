"""Chat loop, slash-command dispatch, and tab completion."""

import sys
from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from lilbee import settings
from lilbee.cli.helpers import (
    add_paths,
    get_version,
    perform_reset,
    render_status,
    stream_response,
)
from lilbee.config import cfg

_ADD_PREFIX = "/add "
_MODEL_PREFIX = "/model "
_VISION_PREFIX = "/vision "


class QuitChat(Exception):
    """Raised by /quit to exit the chat loop."""


def _pick_from_catalog(
    catalog: tuple,
    display_fn: Callable,
    con: Console,
    config_attr: str,
    setting_key: str,
    label: str,
) -> None:
    """Shared interactive picker flow for model/vision catalog selection.

    *display_fn* is called with (ram_gb, free_disk_gb) and returns the recommended ModelInfo.
    *config_attr* is the cfg attribute to set (e.g. "chat_model" or "vision_model").
    *setting_key* is the settings.toml key (e.g. "chat_model" or "vision_model").
    *label* is a human-readable label for messages (e.g. "Switched to model").
    """
    from lilbee.models import (
        get_free_disk_gb,
        get_system_ram_gb,
        pull_with_progress,
        validate_disk_and_pull,
    )

    ram_gb = get_system_ram_gb()
    free_disk_gb = get_free_disk_gb(cfg.data_dir)
    recommended = display_fn(ram_gb, free_disk_gb)
    default_idx = list(catalog).index(recommended) + 1
    installed = set(list_ollama_models())

    try:
        raw = input(f"Choice [{default_idx}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not raw:
        return

    try:
        choice = int(raw)
    except ValueError:
        con.print(f"[red]Enter a number 1-{len(catalog)}.[/red]")
        return

    if not (1 <= choice <= len(catalog)):
        con.print(f"[red]Enter a number 1-{len(catalog)}.[/red]")
        return

    model_info = catalog[choice - 1]
    if model_info.name not in installed:
        if config_attr == "chat_model":
            validate_disk_and_pull(model_info, free_disk_gb)
        else:
            pull_with_progress(model_info.name)
    setattr(cfg, config_attr, model_info.name)
    settings.set_value(cfg.data_root, setting_key, model_info.name)
    con.print(f"{label} [bold]{model_info.name}[/bold] (saved)")


def _set_named_model(
    name: str,
    con: Console,
    config_attr: str,
    setting_key: str,
    label: str,
) -> None:
    """Validate and set a named model directly (no picker)."""
    from lilbee.models import ensure_tag

    name = ensure_tag(name)
    available = list_ollama_models()
    if available and name not in available:
        con.print(f"[red]Unknown model:[/red] {name}")
        con.print(f"Available: {', '.join(sorted(available))}")
        return
    setattr(cfg, config_attr, name)
    settings.set_value(cfg.data_root, setting_key, name)
    con.print(f"{label} [bold]{name}[/bold] (saved)")


def handle_slash_status(args: str, con: Console) -> None:
    render_status(con)


def handle_slash_add(args: str, con: Console) -> None:
    raw = args.strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.exists():
            con.print(f"[red]Path not found:[/red] {raw}")
            return
        add_paths([p], con, force=True)
    else:
        try:
            from prompt_toolkit import prompt as pt_prompt
            from prompt_toolkit.completion import PathCompleter

            raw = pt_prompt("path: ", completer=PathCompleter(expanduser=True))
        except (ImportError, EOFError, KeyboardInterrupt):
            return
        raw = raw.strip()
        if not raw:
            return
        p = Path(raw).expanduser()
        if not p.exists():
            con.print(f"[red]Path not found:[/red] {raw}")
            return
        add_paths([p], con, force=True)


def handle_slash_quit(args: str, con: Console) -> None:
    raise QuitChat


def handle_slash_model(args: str, con: Console) -> None:
    name = args.strip()
    if name:
        _set_named_model(name, con, "chat_model", "chat_model", "Switched to model")
        return

    con.print(f"[bold]Current model:[/bold] {cfg.chat_model}\n")
    from lilbee.models import MODEL_CATALOG, display_model_picker

    _pick_from_catalog(
        MODEL_CATALOG,
        display_model_picker,
        con,
        config_attr="chat_model",
        setting_key="chat_model",
        label="Switched to model",
    )


def handle_slash_vision(args: str, con: Console) -> None:
    name = args.strip()

    # /vision off — disable vision OCR
    if name == "off":
        cfg.vision_model = ""
        settings.set_value(cfg.data_root, "vision_model", "")
        con.print("Vision OCR [bold]disabled[/bold] (saved)")
        return

    # /vision <name> — switch directly
    if name:
        _set_named_model(name, con, "vision_model", "vision_model", "Vision model set to")
        return

    # /vision — show picker
    current = cfg.vision_model
    if current:
        con.print(f"[bold]Current vision model:[/bold] {current}\n")
    else:
        con.print("[bold]Vision OCR:[/bold] disabled\n")

    from lilbee.models import VISION_CATALOG, display_vision_picker

    _pick_from_catalog(
        VISION_CATALOG,
        display_vision_picker,
        con,
        config_attr="vision_model",
        setting_key="vision_model",
        label="Vision model set to",
    )


def handle_slash_version(args: str, con: Console) -> None:
    con.print(f"lilbee [bold]{get_version()}[/bold]")


def handle_slash_reset(args: str, con: Console) -> None:
    con.print(
        "[bold red]This will delete ALL documents and data.[/bold red]\nType 'yes' to confirm:"
    )
    try:
        answer = con.input("[bold red]> [/bold red]")
    except (EOFError, KeyboardInterrupt):
        con.print("Aborted.")
        return
    if answer.strip().lower() != "yes":
        con.print("Aborted.")
        return
    result = perform_reset()
    con.print(
        f"Reset complete: {result['deleted_docs']} document(s), "
        f"{result['deleted_data']} data item(s) deleted."
    )


def handle_slash_help(args: str, con: Console) -> None:
    con.print("[bold]Slash commands:[/bold]")
    con.print("  /status  — show indexed documents and config")
    con.print("  /add [path]  — add a file or directory (tab-completes without args)")
    con.print("  /model [name]  — show or switch chat model")
    con.print("  /vision [name|off]  — show or switch vision OCR model")
    con.print("  /version — show lilbee version")
    con.print("  /reset   — delete all documents and data")
    con.print("  /help    — show this help")
    con.print("  /quit    — exit chat")


_SLASH_COMMANDS: dict[str, Callable[[str, Console], None]] = {
    "status": handle_slash_status,
    "add": handle_slash_add,
    "model": handle_slash_model,
    "vision": handle_slash_vision,
    "version": handle_slash_version,
    "reset": handle_slash_reset,
    "help": handle_slash_help,
    "quit": handle_slash_quit,
}


def dispatch_slash(raw_input: str, con: Console) -> bool:
    """Try to dispatch *raw_input* as a ``/command``.  Returns True if handled."""
    stripped = raw_input.strip()
    if not stripped.startswith("/"):
        return False
    parts = stripped[1:].split(None, 1)
    cmd = parts[0].lower() if parts else ""
    args = parts[1] if len(parts) > 1 else ""
    handler = _SLASH_COMMANDS.get(cmd)
    if handler is None:
        con.print(f"[red]Unknown command:[/red] /{cmd}  — try /help")
        return True
    handler(args, con)
    return True


def list_ollama_models() -> list[str]:
    """Return installed Ollama model names with explicit tags, excluding embedding models."""
    try:
        import ollama

        embed_base = cfg.embedding_model.split(":")[0]
        return [
            m.model for m in ollama.list().models if m.model and m.model.split(":")[0] != embed_base
        ]
    except Exception:
        return []


def make_completer():  # type: ignore[no-untyped-def]
    """Build a completer class that inherits from prompt_toolkit.completion.Completer."""
    from prompt_toolkit.completion import Completer, Completion, PathCompleter
    from prompt_toolkit.document import Document

    class LilbeeCompleter(Completer):
        def get_completions(self, document, complete_event):  # type: ignore[no-untyped-def,override]
            text = document.text_before_cursor
            if text.startswith(_ADD_PREFIX):
                sub_text = text[len(_ADD_PREFIX) :]
                sub_doc = Document(sub_text, len(sub_text))
                yield from PathCompleter(expanduser=True).get_completions(sub_doc, complete_event)
            elif text.startswith(_MODEL_PREFIX):
                prefix = text[len(_MODEL_PREFIX) :]
                for name in list_ollama_models():
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))
            elif text.startswith(_VISION_PREFIX):
                from lilbee.models import VISION_CATALOG

                prefix = text[len(_VISION_PREFIX) :]
                for model in VISION_CATALOG:
                    if model.name.startswith(prefix):
                        yield Completion(model.name, start_position=-len(prefix))
                if "off".startswith(prefix):
                    yield Completion("off", start_position=-len(prefix))
            elif text.startswith("/"):
                prefix = text[1:]
                for cmd in _SLASH_COMMANDS:
                    if cmd.startswith(prefix):
                        yield Completion(f"/{cmd}", start_position=-len(text))

    return LilbeeCompleter()


def chat_loop(con: Console) -> None:
    """Interactive REPL with slash-command support."""
    con.print("[bold]lilbee chat[/bold] — type /help for commands\n")
    history: list[dict] = []

    _prompt_fn: Callable[[], str] | None = None
    if sys.stdin.isatty():
        try:
            from prompt_toolkit import PromptSession

            _session: PromptSession[str] = PromptSession(completer=make_completer())
            _prompt_fn = lambda: _session.prompt("> ")  # noqa: E731
        except ImportError:
            pass

    while True:
        try:
            if _prompt_fn is not None:
                question = _prompt_fn()
            else:
                question = con.input("[bold green]> [/bold green]")
        except (EOFError, KeyboardInterrupt):
            break
        if not question.strip():
            continue
        try:
            if dispatch_slash(question, con):
                continue
        except QuitChat:
            break
        stream_response(question, history, con)
