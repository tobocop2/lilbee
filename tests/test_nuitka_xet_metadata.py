"""The frozen binary's Xet support hangs on one exact string.

huggingface_hub asks importlib.metadata for "hf_xet" and never imports the
module, so the binary needs metadata registered under that spelling. Only a
package configuration can do it: --include-distribution-metadata rewrites the
name to the distribution's canonical "hf-xet", and Nuitka's frozen lookup folds
case but not '-' against '_'.
"""

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "tools" / "wheel-build" / "lilbee.nuitka-package.config.yml"
_BUILD = _ROOT / "tools" / "wheel-build" / "build_lilbee_binary.sh"


def _declared_metadata_names() -> list[str]:
    entries = yaml.safe_load(_CONFIG.read_text())
    return [
        name
        for entry in entries
        for data_file in entry.get("data-files", [])
        for name in data_file.get("include-metadata", [])
    ]


def test_declared_name_is_the_one_huggingface_hub_queries() -> None:
    """A canonical "hf-xet" here would build clean and disable xet silently."""
    from huggingface_hub.utils import _runtime

    assert _runtime._CANDIDATES["hf_xet"] <= set(_declared_metadata_names())


def test_trigger_module_is_the_one_that_reads_the_metadata() -> None:
    """Nuitka applies the config only when it compiles the named module, so the
    trigger has to be the module whose import populates the version table."""
    from huggingface_hub.utils import _runtime

    entries = yaml.safe_load(_CONFIG.read_text())
    modules = {entry["module-name"] for entry in entries}

    assert _runtime.__name__ in modules
    assert _runtime._package_versions["hf_xet"] == _runtime._get_version("hf_xet")


def test_build_passes_the_config_and_compiles_hf_xet() -> None:
    """Metadata is only emitted for a distribution whose top-level module is compiled."""
    build = _BUILD.read_text()

    assert f"--user-package-configuration-file=tools/wheel-build/{_CONFIG.name}" in build
    assert "--include-package=hf_xet" in build
