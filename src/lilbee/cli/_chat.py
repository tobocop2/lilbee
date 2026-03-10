"""Chat loop, slash-command dispatch, and tab completion."""

import sys
from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from lilbee.cli._helpers import (
    _add_paths,
    _get_version,
    _perform_reset,
    _render_status,
    _stream_response,
)

_ADD_PREFIX = "/add "
_MODEL_PREFIX = "/model "


class _QuitChat(Exception):
    """Raised by /quit to exit the chat loop."""


def _handle_slash_status(args: str, con: Console) -> None:
    _render_status(con)


def _handle_slash_add(args: str, con: Console) -> None:
    raw = args.strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.exists():
            con.print(f"[red]Path not found:[/red] {raw}")
            return
        _add_paths([p], con, force=True)
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
        _add_paths([p], con, force=True)


def _handle_slash_quit(args: str, con: Console) -> None:
    raise _QuitChat


def _handle_slash_model(args: str, con: Console) -> None:
    import lilbee.config as cfg
    from lilbee.models import (
        MODEL_CATALOG,
        _validate_disk_and_pull,
        display_model_picker,
        get_free_disk_gb,
        get_system_ram_gb,
    )

    name = args.strip()
    if not name:
        con.print(f"[bold]Current model:[/bold] {cfg.CHAT_MODEL}\n")
        ram_gb = get_system_ram_gb()
        free_disk_gb = get_free_disk_gb(cfg.DATA_DIR)
        recommended = display_model_picker(ram_gb, free_disk_gb)
        default_idx = list(MODEL_CATALOG).index(recommended) + 1
        installed = set(_list_ollama_models())

        try:
            raw = input(f"Choice [{default_idx}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return

        if not raw:
            return

        try:
            choice = int(raw)
        except ValueError:
            con.print(f"[red]Enter a number 1-{len(MODEL_CATALOG)}.[/red]")
            return

        if not (1 <= choice <= len(MODEL_CATALOG)):
            con.print(f"[red]Enter a number 1-{len(MODEL_CATALOG)}.[/red]")
            return

        model_info = MODEL_CATALOG[choice - 1]
        if model_info.name in installed:
            cfg.CHAT_MODEL = model_info.name
            from lilbee import settings

            settings.set_value("chat_model", model_info.name)
            con.print(f"Switched to model [bold]{model_info.name}[/bold] (saved)")
        else:
            _validate_disk_and_pull(model_info, free_disk_gb)
            con.print(f"Switched to model [bold]{model_info.name}[/bold] (saved)")
        return
    available = _list_ollama_models()
    if available and name not in available:
        con.print(f"[red]Unknown model:[/red] {name}")
        con.print(f"Available: {', '.join(sorted(available))}")
        return
    cfg.CHAT_MODEL = name
    from lilbee import settings

    settings.set_value("chat_model", name)
    con.print(f"Switched to model [bold]{name}[/bold] (saved)")


def _handle_slash_version(args: str, con: Console) -> None:
    con.print(f"lilbee [bold]{_get_version()}[/bold]")


def _handle_slash_reset(args: str, con: Console) -> None:
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
    result = _perform_reset()
    con.print(
        f"Reset complete: {result['deleted_docs']} document(s), "
        f"{result['deleted_data']} data item(s) deleted."
    )


def _handle_slash_help(args: str, con: Console) -> None:
    con.print("[bold]Slash commands:[/bold]")
    con.print("  /status  — show indexed documents and config")
    con.print("  /add [path]  — add a file or directory (tab-completes without args)")
    con.print("  /model [name]  — show or switch chat model")
    con.print("  /version — show lilbee version")
    con.print("  /reset   — delete all documents and data")
    con.print("  /help    — show this help")
    con.print("  /quit    — exit chat")


_SLASH_COMMANDS: dict[str, Callable[[str, Console], None]] = {
    "status": _handle_slash_status,
    "add": _handle_slash_add,
    "model": _handle_slash_model,
    "version": _handle_slash_version,
    "reset": _handle_slash_reset,
    "help": _handle_slash_help,
    "quit": _handle_slash_quit,
}


def _dispatch_slash(raw_input: str, con: Console) -> bool:
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


def _list_ollama_models() -> list[str]:
    """Return installed Ollama chat model names, excluding embedding models."""
    try:
        import ollama

        import lilbee.config as cfg

        embed_base = cfg.EMBEDDING_MODEL.split(":")[0]
        return [
            m.model for m in ollama.list().models if m.model and m.model.split(":")[0] != embed_base
        ]
    except Exception:
        return []


def _make_completer():  # type: ignore[no-untyped-def]
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
                for name in _list_ollama_models():
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))
            elif text.startswith("/"):
                prefix = text[1:]
                for cmd in _SLASH_COMMANDS:
                    if cmd.startswith(prefix):
                        yield Completion(f"/{cmd}", start_position=-len(text))

    return LilbeeCompleter()


def _chat_loop(con: Console) -> None:
    """Interactive REPL with slash-command support."""
    con.print("[bold]lilbee chat[/bold] — type /help for commands\n")
    history: list[dict] = []

    _prompt_fn: Callable[[], str] | None = None
    if sys.stdin.isatty():
        try:
            from prompt_toolkit import PromptSession

            _session: PromptSession[str] = PromptSession(completer=_make_completer())
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
            if _dispatch_slash(question, con):
                continue
        except _QuitChat:
            break
        _stream_response(question, history, con)
