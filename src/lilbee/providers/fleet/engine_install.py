"""Install advice for a missing engine, in the form that fits how lilbee was installed."""

from __future__ import annotations

from lilbee._frozen import is_frozen

_INSTALL_COMMAND = (
    "To download the engine, run 'lilbee engine install' (or POST /api/engines/install). "
    "To use a pack on a computer with no network, copy it into the packs folder "
    "of the engines folder first."
)
_NO_PACK = "No engine pack is published for this computer yet, so there is nothing to download."
_BUNDLED_FORMS = (
    "Install a self-contained build again; it carries its own engine: "
    "'brew reinstall lilbee' (Homebrew), 'npm install -g lilbee' (npm launcher), "
    "or the standalone binary from https://github.com/tobocop2/lilbee/releases/latest."
)
_BRING_YOUR_OWN = (
    "To bring your own engine, set LILBEE_LLAMA_SERVER_PATH to a llama-server binary "
    "and put llama-swap / gguf-parser on PATH."
)


def engine_install_hint(bundled: bool | None = None) -> str:
    """The advice for installing the engine, for a self-contained build or a package install.

    *bundled* defaults to whether this lilbee is a frozen, self-contained build.
    """
    bundled = is_frozen() if bundled is None else bundled
    if bundled:
        return f"{_BUNDLED_FORMS} {_BRING_YOUR_OWN}"
    return f"{_download_advice()} {_BUNDLED_FORMS} {_BRING_YOUR_OWN}"


def _download_advice() -> str:
    """The install command, or the reason there is no pack to download for this computer."""
    from lilbee.providers.fleet.engines.choice import PinError
    from lilbee.providers.fleet.first_use import INSTALL_REFUSALS, PackNotPublishedError, pack_lock

    try:
        pack_lock()
    except PackNotPublishedError:
        return _NO_PACK
    except (*INSTALL_REFUSALS, PinError):
        pass  # the install names the exact refusal; the hint keeps the command
    return _INSTALL_COMMAND
