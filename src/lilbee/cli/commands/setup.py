"""Token (server auth), HuggingFace login, self-check, and crawler-setup commands."""

from __future__ import annotations

import asyncio
import importlib
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

_SELF_CHECK_CHAT_REPO = "Qwen/Qwen3-0.6B-GGUF"
_SELF_CHECK_CHAT_FILE = "Qwen3-0.6B-Q8_0.gguf"
_SELF_CHECK_EMBED_REPO = "nomic-ai/nomic-embed-text-v1.5-GGUF"
_SELF_CHECK_EMBED_FILE = "nomic-embed-text-v1.5.Q4_K_M.gguf"


def _download_self_check_model(repo: str, filename: str) -> Path:
    """Fetch a GGUF from the HuggingFace CDN via urllib (stdlib only).

    Avoids huggingface_hub / httpx entirely. Inside the Nuitka --onefile
    binary, huggingface_hub's retry path has re-entered a closed httpx client
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


def _resolved_provider_kwargs() -> dict[str, Any]:
    """Snapshot of the provider-stack knobs self-check exercises.

    Echoed back in the JSON payload + human readout so users can confirm
    which dynamic ctx / FA / KV cache / GPU layers values their install
    chose without grepping debug logs.
    """
    return {
        "num_ctx": cfg.num_ctx,
        "num_ctx_max": cfg.num_ctx_max,
        "flash_attention": cfg.flash_attention,
        "kv_cache_type": cfg.kv_cache_type.value,
        "n_gpu_layers": cfg.n_gpu_layers,
        "main_gpu": cfg.main_gpu,
        "gpu_devices": cfg.gpu_devices,
    }


def self_check_cmd(
    chat_model_path: Path | None = _self_check_chat_path_option,
    embed_model_path: Path | None = _self_check_embed_path_option,
    max_tokens: int = _self_check_max_tokens_option,
    skip_embedding: bool = _self_check_skip_embedding_option,
) -> None:
    """Verify the installation can load llama.cpp and run real inference.

    Routes both legs through :func:`lilbee.providers.llama_cpp.provider.load_llama`
    so the dynamic-``n_ctx`` picker, flash-attention default, KV cache type,
    ``n_gpu_layers`` resolution, and OOM retry path all run -- i.e. the same
    provider stack a real ``lilbee ask`` / ``lilbee chat`` exercises. Failure
    here means either the vendored shared libraries don't load or one of the
    cfg-driven provider knobs is misconfigured for the host.

    Two legs:

    1. **Chat**: downloads ``Qwen3-0.6B-Q8_0.gguf`` (~500MB),
       runs ``load_llama(..., mode=LoaderMode.CHAT)`` so the dynamic-ctx picker /
       flash-attention default / KV cache mapping fire, then issues a tiny
       ``create_completion``.
    2. **Embedding**: downloads ``nomic-embed-text-v1.5.Q4_K_M.gguf`` (~84MB),
       runs ``load_llama(..., mode=LoaderMode.EMBED)`` so the embed-mode ctx clamp
       fires, then issues ``create_embedding``. Catches the "Memory is not
       initialized" assert from llama-cpp-python <0.3.19, where BERT-style
       encoders trip ``kv_cache_clear`` on a context that never allocated
       memory.

    Exits 0 on success, 1 on any failure. Intended for post-install
    verification and as the end-to-end gate in release CI.
    """
    from typing import cast

    from lilbee.providers.llama_cpp.provider import load_llama
    from lilbee.providers.model_cache import LoaderMode

    try:
        chat_path = chat_model_path or _download_self_check_model(
            _SELF_CHECK_CHAT_REPO, _SELF_CHECK_CHAT_FILE
        )
        console.print(f"Loading chat model {chat_path}")

        llm = load_llama(chat_path, mode=LoaderMode.CHAT)
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
            enc = load_llama(embed_path, mode=LoaderMode.EMBED)
            emb = cast(Any, enc.create_embedding(input=["test"]))
            vec = emb["data"][0]["embedding"]
        except Exception as exc:
            _self_check_emit_failure(repr(exc))
            raise typer.Exit(1) from exc

        if not vec:
            _self_check_emit_failure("empty embedding vector")
            raise typer.Exit(1)
        embedding_dims = len(vec)

    provider_kwargs = _resolved_provider_kwargs()
    if cfg.json_mode:
        payload: dict[str, Any] = {
            "ok": True,
            "chat_response": text,
            "chat_model": str(chat_path),
            "provider": provider_kwargs,
        }
        if embedding_dims is not None:
            payload["embedding_dims"] = embedding_dims
        json_output(payload)
    else:
        console.print(f"Chat response: {text!r}")
        if embedding_dims is not None:
            console.print(f"Embedding dims: {embedding_dims}")
        console.print(
            f"Provider: num_ctx={provider_kwargs['num_ctx']} "
            f"num_ctx_max={provider_kwargs['num_ctx_max']} "
            f"flash_attention={provider_kwargs['flash_attention']} "
            f"kv_cache_type={provider_kwargs['kv_cache_type']} "
            f"n_gpu_layers={provider_kwargs['n_gpu_layers']} "
            f"main_gpu={provider_kwargs['main_gpu']} "
            f"gpu_devices={provider_kwargs['gpu_devices']}"
        )
        console.print(f"[{theme.ACCENT}]SELF-CHECK PASSED[/{theme.ACCENT}]")


_SELF_CHECK_EXTRAS = ("litellm", "crawl4ai", "spacy", "graspologic_native")


def self_check_extras_cmd() -> None:
    """Verify optional extras (crawler, litellm, graph) are bundled and importable."""
    results: dict[str, Any] = {}
    failed: list[str] = []
    for name in _SELF_CHECK_EXTRAS:
        try:
            importlib.import_module(name)
            results[name] = True
        except ImportError as exc:
            results[name] = False
            results[f"{name}_error"] = str(exc)
            failed.append(name)

    if cfg.json_mode:
        json_output({"ok": not failed, **results})
    else:
        for name in _SELF_CHECK_EXTRAS:
            ok = results.get(name) is True
            tag = (
                f"[{theme.ACCENT}]ok[/{theme.ACCENT}]"
                if ok
                else f"[{theme.ERROR}]MISSING[/{theme.ERROR}]"
            )
            console.print(f"  {name}: {tag}")
            if not ok:
                console.print(f"    {results.get(f'{name}_error', '')}")

    if failed:
        raise typer.Exit(1)


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
