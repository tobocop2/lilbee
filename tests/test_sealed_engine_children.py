"""A child process a test spawns finds no engine on the host, as the test process finds none.

The inner run has a llama-server on ``PATH`` or in ``LILBEE_LLAMA_SERVER_PATH`` before the
suite's fixtures start, as a developer's computer has.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PLANTED = "PLANTED_HOST_ENGINE"
_HELPER = "PLANTED_HOST_HELPER"

_INNER_TESTS = '''
import os
import subprocess
import sys

import pytest

_CHILD = (
    "import shutil\\n"
    "from lilbee.providers.base import ProviderError\\n"
    "from lilbee.providers.fleet.binary import EngineTool, resolve_engine_tool\\n"
    "print(shutil.which('llama-server'))\\n"
    "print(shutil.which('host-helper'))\\n"
    "try:\\n"
    "    print(resolve_engine_tool(EngineTool.LLAMA_SERVER, install=False))\\n"
    "except ProviderError:\\n"
    "    print('NOT_FOUND')\\n"
)


def _child_answers():
    """What a fresh interpreter reports: both lookups and the resolved path."""
    result = subprocess.run(
        [sys.executable, "-c", _CHILD], capture_output=True, encoding="utf-8", check=True
    )
    which, helper, resolved = result.stdout.strip().splitlines()
    return which, helper, resolved


def test_a_child_finds_no_engine_the_host_has():
    which, helper, resolved = _child_answers()
    assert which == "None"
    assert os.environ["PLANTED_HOST_ENGINE"] not in resolved


def test_a_child_still_finds_the_other_tools_of_the_host_folder():
    which, helper, resolved = _child_answers()
    assert (helper != "None") == ("PLANTED_HOST_HELPER" in os.environ)


@pytest.mark.real_engine_resolution
def test_the_unsealed_child_finds_the_planted_engine():
    which, helper, resolved = _child_answers()
    assert os.environ["PLANTED_HOST_ENGINE"] in which + resolved
'''


@pytest.mark.skipif(sys.platform == "win32", reason="the planted engine is a shell script")
@pytest.mark.parametrize("how", ["path", "relative-path", "env"])
def test_a_planted_host_engine_reaches_no_child(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    engine_dir = pytester.mkdir("host-bin")
    engine = engine_dir / "llama-server"
    for tool in (engine, engine_dir / "host-helper"):
        tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        tool.chmod(0o755)
    monkeypatch.setenv("PYTHONPATH", str(_REPO_ROOT))
    if how == "env":
        monkeypatch.setenv(_PLANTED, str(engine))
        monkeypatch.setenv("LILBEE_LLAMA_SERVER_PATH", str(engine))
    else:
        folder = "host-bin" if how == "relative-path" else str(engine_dir)
        monkeypatch.setenv(_PLANTED, folder)
        monkeypatch.setenv(_HELPER, "1")
        monkeypatch.setenv("PATH", os.pathsep.join([folder, os.environ.get("PATH", "")]))
    pytester.makeini("[pytest]\nmarkers =\n    real_engine_resolution: unseal engine resolution\n")
    pytester.makepyfile(test_inner=_INNER_TESTS)

    result = pytester.runpytest_subprocess("-p", "tests.conftest", "-p", "no:cacheprovider")

    result.assert_outcomes(passed=3)
