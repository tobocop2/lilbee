"""Token (server auth), HuggingFace login, self-check, and crawler-setup commands."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import shutil
import signal
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, TypeVar

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

_LegResultT = TypeVar("_LegResultT")

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
    # Any exit other than a successful return drops the temp dir, so a failed
    # download never leaves an empty/partial dir behind.
    try:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310  literal https url
                    dest.write_bytes(response.read())
                return dest
            except (OSError, urllib.error.URLError) as exc:
                last_exc = exc
                console.print(f"download attempt {attempt + 1} failed: {exc!r}")
        raise RuntimeError(f"GGUF download failed after 3 attempts: {last_exc!r}")
    except BaseException:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise


def _installed_model_path(want: str, configured: str) -> Path | None:
    """Path of an installed native GGUF whose task is *want*, or ``None``.

    Prefers the configured ref when it is installed and matches the role,
    so the check exercises the model the user actually runs.
    """
    from lilbee.catalog.query import reclassify_by_name
    from lilbee.modelhub.registry import ModelRegistry

    registry = ModelRegistry(cfg.models_dir)
    try:
        manifests = [
            m for m in registry.list_installed() if reclassify_by_name(m.ref, m.task) == want
        ]
    except Exception:
        return None
    refs = [m.ref for m in manifests]
    ordered = [configured, *refs] if configured in refs else refs
    for ref in ordered:
        try:
            return registry.resolve(ref)
        except (KeyError, ValueError):
            continue
    return None


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
        "cpu_moe": cfg.cpu_moe,
        "n_cpu_moe": cfg.n_cpu_moe,
        "main_gpu": cfg.main_gpu,
        "gpu_devices": cfg.gpu_devices,
    }


def _self_check_server(
    role: WorkerRole, model_path: Path
) -> tuple[SwapManager, LlamaServerClient, Path]:
    """Start a one-model llama-swap for *model_path* in *role* and return its
    manager plus an OpenAI client.

    Asks the planner for the launch it would build rather than assembling one
    beside it, so the check exercises the slots, context, pinning and flags a
    real request drives. The upstream loads on the first request; the caller
    shuts the manager down.
    """
    from lilbee.providers.fleet.client import LlamaServerClient
    from lilbee.providers.fleet.groups import SwapGroup
    from lilbee.providers.fleet.planning import build_single_role_launch
    from lilbee.providers.fleet.swap_manager import SwapManager

    launch = build_single_role_launch(role, model_path)
    work_dir = Path(tempfile.mkdtemp(prefix="lilbee-self-check-"))
    swap = SwapManager(work_dir, SwapGroup(role.value))
    try:
        swap.start([launch])
    except BaseException:
        # start() raises on engine-load failure (the case self-check exists to
        # catch); work_dir is never returned, so clean it here rather than orphan it.
        swap.shutdown()
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    client = LlamaServerClient(
        swap.endpoint(), launch.model_id, inline_reasoning=role is WorkerRole.CHAT
    )
    return swap, client, work_dir


def _self_check_chat(model_path: Path, max_tokens: int) -> str:
    """Run a chat model through a one-off llama-swap, request a tiny completion, tear down."""
    swap, client, work_dir = _self_check_server(WorkerRole.CHAT, model_path)
    try:
        result = client.chat(
            [{"role": "user", "content": "2+2="}],
            options={"max_tokens": max_tokens},
            stream=False,
        )
        return str(result)
    finally:
        swap.shutdown()
        shutil.rmtree(work_dir, ignore_errors=True)


def _self_check_embed(model_path: Path) -> int:
    """Run an embedding model through a one-off llama-swap, return one vector's dim."""
    swap, client, work_dir = _self_check_server(WorkerRole.EMBED, model_path)
    try:
        vectors = client.embed(["test"])
        return len(vectors[0]) if vectors else 0
    finally:
        swap.shutdown()
        shutil.rmtree(work_dir, ignore_errors=True)


def _self_check_leg(
    model_path: Path | None,
    repo: str,
    filename: str,
    label: str,
    check: Callable[[Path], _LegResultT],
) -> tuple[_LegResultT, Path]:
    """Resolve a model (user path or download), run *check*, and clean any download.

    On any failure emits the structured failure and exits 1, matching the
    per-leg error handling the self-check command used inline.
    """
    download_dir: Path | None = None
    try:
        if model_path is None:
            model_path = _download_self_check_model(repo, filename)
            download_dir = model_path.parent
        console.print(f"Loading {label} model {model_path}")
        result = check(model_path)
    except Exception as exc:
        _self_check_emit_failure(repr(exc))
        raise typer.Exit(1) from exc
    finally:
        if download_dir is not None:
            shutil.rmtree(download_dir, ignore_errors=True)
    return result, model_path


@contextlib.contextmanager
def _teardown_on_sigterm() -> Iterator[None]:
    """Convert SIGTERM into an exception so the self-check teardown runs.

    Each leg tears its fleet down and removes its temp dir in a ``finally``. The
    default SIGTERM disposition ends the interpreter without unwinding, orphaning
    the engine; raising instead runs the same cleanup a ctrl-c (SIGINT) does.
    No-op off the main thread and where SIGTERM is not delivered (Windows).
    """

    def _raise(_signum: int, _frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    try:
        previous = signal.signal(signal.SIGTERM, _raise)
    except ValueError:  # pragma: no cover - not the main thread
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


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

    Two legs, each preferring an already-installed model of the role. Only when
    nothing suitable is installed does a leg download a pinned tiny model to a
    temp dir, removed when the leg finishes:

    1. **Chat**: ``Qwen3-0.6B-Q8_0.gguf`` (~500MB), spawns a chat server, and
       requests a tiny completion.
    2. **Embedding**: ``nomic-embed-text-v1.5.Q4_K_M.gguf`` (~84MB), spawns an
       embedding server, and requests one embedding vector.

    Exits 0 on success, 1 on any failure. Intended for post-install
    verification and as the end-to-end gate in release CI.
    """
    from lilbee.catalog.types import ModelTask

    if chat_model_path is None:
        chat_model_path = _installed_model_path(ModelTask.CHAT, cfg.chat_model)
    if embed_model_path is None and not skip_embedding:
        embed_model_path = _installed_model_path(ModelTask.EMBEDDING, cfg.embedding_model)

    with _teardown_on_sigterm():
        text, chat_path = _self_check_leg(
            chat_model_path,
            _SELF_CHECK_CHAT_REPO,
            _SELF_CHECK_CHAT_FILE,
            "chat",
            lambda p: _self_check_chat(p, max_tokens),
        )

        if not text.strip():
            _self_check_emit_failure("empty inference response")
            raise typer.Exit(1)

        embedding_dims: int | None = None
        if not skip_embedding:
            embedding_dims, _ = _self_check_leg(
                embed_model_path,
                _SELF_CHECK_EMBED_REPO,
                _SELF_CHECK_EMBED_FILE,
                "embedding",
                _self_check_embed,
            )

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

# The name of the functional charset-detection leg in self-check-extras output.
_CHARSET_PROBE = "charset_detection"


def _probe_charset_detection() -> str | None:
    """Run the real chardet pipeline; return an error string when it is broken.

    A bare `import chardet` passes even when the frozen bundle is broken:
    chardet imports `chardet.models` (and loads its `.bin` data) lazily on the
    first `detect()` call, and the crawl path only reaches that call for a
    response without a charset header. This probe forces the full detection
    pipeline offline, so the release gate fails deterministically when a
    frozen build cannot detect charsets.
    """
    # Any failure below means the bundle is broken; the caller reports the
    # error and fails the check, so nothing is swallowed.
    try:
        import chardet

        sample = "字符集检测自检文本 用于验证冻结构建" * 8
        result = chardet.detect(sample.encode("gb18030"))
        if not result.get("encoding"):
            return f"chardet.detect returned no encoding: {result!r}"
        return None
    except Exception as exc:
        return str(exc)


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

    probe_error = _probe_charset_detection()
    results[_CHARSET_PROBE] = probe_error is None
    if probe_error is not None:
        results[f"{_CHARSET_PROBE}_error"] = probe_error
        failed.append(_CHARSET_PROBE)

    if cfg.json_mode:
        json_output({"ok": not failed, **results})
    else:
        for name in (*_SELF_CHECK_EXTRAS, _CHARSET_PROBE):
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
        data = json.loads(path.read_text(encoding="utf-8"))
        tok = data.get("token", "")
    # UnicodeDecodeError is a ValueError, not a JSONDecodeError.
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
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
