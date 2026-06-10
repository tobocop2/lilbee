"""Token (server auth), HuggingFace login, self-check, and crawler-setup commands."""

from __future__ import annotations

import asyncio
import importlib
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from lilbee.providers.roles import WorkerRole
from lilbee.runtime.progress import EventType, SetupProgressEvent

if TYPE_CHECKING:
    from lilbee.providers.fleet.client import LlamaServerClient
    from lilbee.providers.fleet.swap_manager import SwapManager

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
        "chat_n_ctx_target": cfg.chat_n_ctx_target,
        "flash_attention": cfg.flash_attention,
        "kv_cache_type": cfg.kv_cache_type.value,
        "n_gpu_layers": cfg.n_gpu_layers,
        "main_gpu": cfg.main_gpu,
        "gpu_devices": cfg.gpu_devices,
    }


def _self_check_server(role: WorkerRole, model_path: Path) -> tuple[SwapManager, LlamaServerClient]:
    """Start a one-model llama-swap for *model_path* in *role* and return its
    manager plus an OpenAI client.

    Builds the launch with the fleet's per-role argv builder (ctx, gpu-layers,
    and -- for chat -- the flash-attn / KV-cache flags) so the check exercises the
    same binary and flags a real request drives. The upstream loads on the first
    request; the caller shuts the manager down.
    """
    from lilbee.core.config.enums import KvCacheType
    from lilbee.providers.engine_params import (
        resolve_chat_ctx,
        resolve_embed_ctx,
        resolve_n_gpu_layers,
    )
    from lilbee.providers.fleet.adapters import ROLE_SPECS, build_server_argv
    from lilbee.providers.fleet.binary import (
        llama_server_runtime_env,
        resolve_llama_server,
    )
    from lilbee.providers.fleet.client import LlamaServerClient
    from lilbee.providers.fleet.launch import InstanceLaunch
    from lilbee.providers.fleet.swap_manager import SwapManager
    from lilbee.providers.gguf_meta import read_gguf_metadata

    meta = read_gguf_metadata(model_path)
    is_embed = role is WorkerRole.EMBED
    if is_embed:
        ctx = resolve_embed_ctx(meta, model_path)
    else:
        ctx = cfg.num_ctx or resolve_chat_ctx(model_path, meta)
    argv = build_server_argv(
        binary=resolve_llama_server(),
        spec=ROLE_SPECS[role],
        model_path=model_path,
        devices=(),
        n_gpu_layers=resolve_n_gpu_layers(embedding=is_embed),
        slots=1,
        ctx_per_slot=ctx,
        # Chat mirrors the fleet's chat flags; embed runs f16 KV with a full-ctx batch.
        flash_attn=None if is_embed else ("off" if cfg.flash_attention is False else "on"),
        cache_type=(
            None if is_embed or cfg.kv_cache_type is KvCacheType.F16 else cfg.kv_cache_type.value
        ),
        batch_size=ctx if is_embed else None,
    )
    work_dir = Path(tempfile.mkdtemp(prefix="lilbee-self-check-"))
    launch = InstanceLaunch(
        role=role,
        argv=argv,
        env_overrides=llama_server_runtime_env(),
        model=str(model_path),
        token_cap=ctx if is_embed else None,
    )
    swap = SwapManager(work_dir)
    swap.start([launch])
    return swap, LlamaServerClient(swap.endpoint(), launch.model_id)


def _self_check_chat(model_path: Path, max_tokens: int) -> str:
    """Run a chat model through a one-off llama-swap, request a tiny completion, tear down."""
    swap, client = _self_check_server(WorkerRole.CHAT, model_path)
    try:
        result = client.chat(
            [{"role": "user", "content": "2+2="}],
            options={"max_tokens": max_tokens},
            stream=False,
        )
        return str(result)
    finally:
        swap.shutdown()


def _self_check_embed(model_path: Path) -> int:
    """Run an embedding model through a one-off llama-swap, return one vector's dim."""
    swap, client = _self_check_server(WorkerRole.EMBED, model_path)
    try:
        vectors = client.embed(["test"])
        return len(vectors[0]) if vectors else 0
    finally:
        swap.shutdown()


def self_check_cmd(
    chat_model_path: Path | None = _self_check_chat_path_option,
    embed_model_path: Path | None = _self_check_embed_path_option,
    max_tokens: int = _self_check_max_tokens_option,
    skip_embedding: bool = _self_check_skip_embedding_option,
) -> None:
    """Verify the installation can launch llama-server and run real inference.

    Spawns a one-off llama-server for each leg with the same launch builder the
    fleet uses (so the dynamic-``n_ctx`` picker, flash-attention default, KV cache
    type, and ``n_gpu_layers`` resolution all fire), then issues a request over
    HTTP -- i.e. the same engine a real ``lilbee ask`` / ``lilbee chat`` drives.
    Failure here means either the bundled binary / its shared libraries don't load
    or one of the cfg-driven knobs is misconfigured for the host.

    Two legs:

    1. **Chat**: downloads ``Qwen3-0.6B-Q8_0.gguf`` (~500MB), spawns a chat
       server, and requests a tiny completion.
    2. **Embedding**: downloads ``nomic-embed-text-v1.5.Q4_K_M.gguf`` (~84MB),
       spawns an embedding server, and requests one embedding vector.

    Exits 0 on success, 1 on any failure. Intended for post-install
    verification and as the end-to-end gate in release CI.
    """
    try:
        chat_path = chat_model_path or _download_self_check_model(
            _SELF_CHECK_CHAT_REPO, _SELF_CHECK_CHAT_FILE
        )
        console.print(f"Loading chat model {chat_path}")
        text = _self_check_chat(chat_path, max_tokens)
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
            embedding_dims = _self_check_embed(embed_path)
        except Exception as exc:
            _self_check_emit_failure(repr(exc))
            raise typer.Exit(1) from exc

        if not embedding_dims:
            _self_check_emit_failure("empty embedding vector")
            raise typer.Exit(1)

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
            f"chat_n_ctx_target={provider_kwargs['chat_n_ctx_target']} "
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
