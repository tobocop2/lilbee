"""Token (server auth), HuggingFace login, self-check, and crawler-setup commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer

from lilbee.cli import theme
from lilbee.cli.app import (
    apply_overrides,
    console,
    data_dir_option,
    global_option,
)
from lilbee.cli.helpers import json_output
from lilbee.cli.tui import messages as msg
from lilbee.core.config import cfg
from lilbee.crawler import CrawlerBrowserError, bootstrap_chromium, chromium_installed
from lilbee.runtime.progress import EventType, SetupProgressEvent

_SELF_CHECK_CHAT_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
_SELF_CHECK_CHAT_FILE = "SmolLM2-135M-Instruct-Q3_K_S.gguf"
_SELF_CHECK_EMBED_REPO = "nomic-ai/nomic-embed-text-v1.5-GGUF"
_SELF_CHECK_EMBED_FILE = "nomic-embed-text-v1.5.Q4_K_M.gguf"


def _download_self_check_model(repo: str, filename: str) -> Path:
    """Fetch a GGUF from the HuggingFace CDN via urllib (stdlib only).

    Avoids huggingface_hub / httpx entirely. Inside the PyInstaller --onefile
    bundle, huggingface_hub's retry path has re-entered a closed httpx client
    after transient DNS failures on macOS runners. urllib is synchronous,
    lives in the stdlib, and has no long-lived client to close.
    """
    import tempfile
    import urllib.request

    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    dest_dir = Path(tempfile.mkdtemp(prefix="lilbee-self-check-"))
    dest = dest_dir / filename
    console.print(f"Downloading {url}")
    last_exc: BaseException | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310  literal https url
                dest.write_bytes(response.read())
            return dest
        except (OSError, urllib.error.URLError) as exc:
            last_exc = exc
            console.print(f"download attempt {attempt + 1} failed: {exc!r}")
    raise RuntimeError(f"GGUF download failed after 3 attempts: {last_exc!r}")


_self_check_chat_path_option = typer.Option(
    None,
    "--chat-model-path",
    help="Path to a chat GGUF file. Skips the HuggingFace download.",
)
_self_check_embed_path_option = typer.Option(
    None,
    "--embed-model-path",
    help="Path to an embedding GGUF file. Skips the HuggingFace download.",
)
_self_check_max_tokens_option = typer.Option(5, "--max-tokens", help="Tokens to generate.")
_self_check_skip_embedding_option = typer.Option(
    False,
    "--skip-embedding",
    help="Skip the embedding-model leg of the self-check.",
)


def _self_check_emit_failure(error: str) -> None:
    if cfg.json_mode:
        json_output({"ok": False, "error": error})
    else:
        console.print(f"[{theme.ERROR}]SELF-CHECK FAILED:[/{theme.ERROR}] {error}")


def self_check_cmd(
    chat_model_path: Path | None = _self_check_chat_path_option,
    embed_model_path: Path | None = _self_check_embed_path_option,
    max_tokens: int = _self_check_max_tokens_option,
    skip_embedding: bool = _self_check_skip_embedding_option,
) -> None:
    """Verify the installation can load llama.cpp and run real inference.

    Two legs:

    1. **Chat**: downloads ``SmolLM2-135M-Instruct-Q3_K_S.gguf`` (~90MB) and
       runs a tiny ``create_completion`` so we know decoder-style models work
       end-to-end and the vendored shared libraries load.
    2. **Embedding**: downloads ``nomic-embed-text-v1.5.Q4_K_M.gguf`` (~84MB)
       and runs ``create_embedding``. This is the leg that catches the
       "Memory is not initialized" assert from llama-cpp-python <0.3.19, where
       BERT-style encoders trip ``kv_cache_clear`` on a context that never
       allocated memory.

    Exits 0 on success, 1 on any failure. Intended for post-install
    verification and as the end-to-end gate in release CI.
    """
    from typing import cast

    try:
        chat_path = chat_model_path or _download_self_check_model(
            _SELF_CHECK_CHAT_REPO, _SELF_CHECK_CHAT_FILE
        )
        console.print(f"Loading chat model {chat_path}")

        import llama_cpp

        from lilbee.providers.llama_cpp_provider import install_llama_log_handler

        install_llama_log_handler()
        llm = llama_cpp.Llama(model_path=str(chat_path), n_ctx=256, verbose=False)
        # stream=False (default) returns a dict, not an iterator, but
        # create_completion's return type is a union; cast to Any so the
        # indexing below type-checks without forcing llama_cpp to be a
        # typecheck-time dep of lilbee.
        out = cast(Any, llm.create_completion("2+2=", max_tokens=max_tokens))
        text: str = out["choices"][0]["text"]
    except Exception as exc:
        _self_check_emit_failure(repr(exc))
        raise typer.Exit(1) from exc

    if not text.strip():
        _self_check_emit_failure("empty inference response")
        raise typer.Exit(1)

    embedding_dims: int | None = None
    if not skip_embedding:
        try:
            embed_path = embed_model_path or _download_self_check_model(
                _SELF_CHECK_EMBED_REPO, _SELF_CHECK_EMBED_FILE
            )
            console.print(f"Loading embedding model {embed_path}")
            enc = llama_cpp.Llama(
                model_path=str(embed_path),
                embedding=True,
                n_ctx=512,
                verbose=False,
            )
            emb = cast(Any, enc.create_embedding(input=["test"]))
            vec = emb["data"][0]["embedding"]
        except Exception as exc:
            _self_check_emit_failure(repr(exc))
            raise typer.Exit(1) from exc

        if not vec:
            _self_check_emit_failure("empty embedding vector")
            raise typer.Exit(1)
        embedding_dims = len(vec)

    if cfg.json_mode:
        payload: dict[str, Any] = {
            "ok": True,
            "chat_response": text,
            "chat_model": str(chat_path),
        }
        if embedding_dims is not None:
            payload["embedding_dims"] = embedding_dims
        json_output(payload)
    else:
        console.print(f"Chat response: {text!r}")
        if embedding_dims is not None:
            console.print(f"Embedding dims: {embedding_dims}")
        console.print(f"[{theme.ACCENT}]SELF-CHECK PASSED[/{theme.ACCENT}]")


def token(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Print the auth token for a running server."""
    from lilbee.server.auth import server_json_path

    apply_overrides(data_dir=data_dir, use_global=use_global)
    path = server_json_path()
    if not path.exists():
        if cfg.json_mode:
            json_output({"error": "No running server found"})
        else:
            console.print("No running server found (server.json missing).")
        raise SystemExit(1)
    try:
        data = json.loads(path.read_text())
        tok = data.get("token", "")
    except (json.JSONDecodeError, OSError) as exc:
        if cfg.json_mode:
            json_output({"error": f"Could not read server.json: {exc}"})
        else:
            console.print(
                f"[{theme.ERROR}]Error:[/{theme.ERROR}] Could not read server.json: {exc}"
            )
        raise SystemExit(1) from None
    if cfg.json_mode:
        json_output({"token": tok})
        return
    console.print(tok)


def login() -> None:
    """Log in to HuggingFace for access to gated models (Mistral, Llama, etc.)."""
    import webbrowser

    from huggingface_hub import get_token
    from huggingface_hub import login as hf_login

    if get_token():
        typer.echo("Already logged in to HuggingFace.")
        if not typer.confirm("Log in again?", default=False):
            return

    typer.echo("Opening HuggingFace token page in your browser...")
    typer.echo("Create a token with 'Read' access, then paste it below.\n")
    webbrowser.open("https://huggingface.co/settings/tokens")

    token = typer.prompt("Paste your HuggingFace token", hide_input=True)
    if not token.strip():
        typer.echo("No token provided.", err=True)
        raise typer.Exit(1)

    hf_login(token=token.strip(), add_to_git_credential=False)
    typer.echo("Logged in! Gated models (Mistral, Llama, etc.) are now accessible.")


setup_app = typer.Typer(help="One-time setup for optional runtime components.")


@setup_app.command(name="crawler")
def setup_crawler_cmd() -> None:
    """Install Playwright's Chromium browser, needed for /crawl.

    No-op when Chromium is already present. Emits a simple progress
    readout; use '--json' mode on the top-level 'lilbee' command to get
    a single JSON blob with the final install state instead.
    """
    if chromium_installed():
        if cfg.json_mode:
            typer.echo(json.dumps({"component": "chromium", "already_installed": True}))
        else:
            typer.echo("Chromium already installed.")
        return

    last_pct: list[int] = [-1]

    def _on_progress(event_type: object, data: object) -> None:
        if event_type != EventType.SETUP_PROGRESS or not isinstance(data, SetupProgressEvent):
            return
        total = data.total_bytes or 0
        pct = int(data.downloaded_bytes * 100 / total) if total > 0 else 0
        if pct != last_pct[0] and not cfg.json_mode:
            last_pct[0] = pct
            typer.echo(msg.SETUP_CHROMIUM_CLI_PROGRESS.format(pct=pct), err=True)

    try:
        asyncio.run(bootstrap_chromium(on_progress=_on_progress))
    except CrawlerBrowserError as exc:
        if cfg.json_mode:
            typer.echo(json.dumps({"component": "chromium", "error": str(exc)}))
        else:
            typer.secho(f"Install failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    if cfg.json_mode:
        typer.echo(json.dumps({"component": "chromium", "installed": True}))
    else:
        typer.echo("Chromium installed.")
