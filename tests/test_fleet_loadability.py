"""The gguf-parser probe that refuses a GGUF the bundled engine cannot decode."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lilbee.catalog.compat import UnsupportedQuantError
from lilbee.providers.fleet import loadability
from lilbee.providers.fleet.loadability import _named_quant, assert_engine_can_load

_REPO = "prism-ml/Ternary-Bonsai-27B-gguf"
_FILE = "Ternary-Bonsai-27B-Q2_0.gguf"

# The three shapes gguf-parser's output takes, verbatim from the real binary.
_TENSOR_FAILURE = (
    "failed to parse GGUF file: read tensor info 0: "
    "GGMLType(42): This quantized type is currently unsupported"
)
_MISSING_FILE = (
    "failed to parse GGUF file: open http file: stat: do head request: stat: status code 404"
)
_NO_SUCH_HOST = (
    'failed to parse GGUF file: open http file: stat: do head request: do request: Head "x": '
    "lookup nonexistent.invalid: no such host"
)


@pytest.fixture(autouse=True)
def _stub_parser_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The probe must not need a real engine wheel to build its argv."""
    monkeypatch.setattr(loadability, "resolve_gguf_parser", lambda: tmp_path / "gguf-parser")


def _returns(monkeypatch: pytest.MonkeyPatch, output: str, returncode: int) -> list[list[str]]:
    """Stub run_bounded with a fixed result, capturing the argv it was given."""
    calls: list[list[str]] = []

    def _run(argv: list[str], **_kwargs: object) -> tuple[str, int]:
        calls.append(argv)
        return output, returncode

    monkeypatch.setattr(loadability, "run_bounded", _run)
    return calls


class TestVerdicts:
    def test_refuses_a_tensor_type_the_engine_cannot_decode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _returns(monkeypatch, _TENSOR_FAILURE, 1)
        with pytest.raises(UnsupportedQuantError) as excinfo:
            assert_engine_can_load(_REPO, _FILE)
        assert excinfo.value.quant == "GGMLType(42)"
        assert excinfo.value.ref == f"{_REPO}/{_FILE}"
        assert "--allow-unsupported" in str(excinfo.value)

    def test_allows_a_file_the_parser_reads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _returns(monkeypatch, "{}", 0)
        assert assert_engine_can_load(_REPO, _FILE) is None

    @pytest.mark.parametrize(
        ("output", "case"),
        [(_MISSING_FILE, "404"), (_NO_SUCH_HOST, "dns")],
        ids=["missing_file", "no_such_host"],
    )
    def test_an_io_failure_is_not_a_verdict(
        self, monkeypatch: pytest.MonkeyPatch, output: str, case: str
    ) -> None:
        """A repo lilbee cannot reach must not read as an unloadable model."""
        _returns(monkeypatch, output, 1)
        assert assert_engine_can_load(_REPO, _FILE) is None

    def test_a_parser_that_will_not_run_is_not_a_verdict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(argv: list[str], **_kwargs: object) -> tuple[str, int]:
            raise OSError("no such binary")

        monkeypatch.setattr(loadability, "run_bounded", _raise)
        assert assert_engine_can_load(_REPO, _FILE) is None

    def test_a_timeout_is_not_a_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(argv: list[str], **_kwargs: object) -> tuple[str, int]:
            raise subprocess.TimeoutExpired(cmd="gguf-parser", timeout=1.0)

        monkeypatch.setattr(loadability, "run_bounded", _raise)
        assert assert_engine_can_load(_REPO, _FILE) is None


class TestArgv:
    def test_names_the_repo_and_file_and_skips_the_estimate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The probe reads the header only; an estimate would be wasted work."""
        calls = _returns(monkeypatch, "{}", 0)
        assert_engine_can_load(_REPO, _FILE)
        argv = calls[-1]
        assert argv[argv.index("--hf-repo") + 1] == _REPO
        assert argv[argv.index("--hf-file") + 1] == _FILE
        assert "--skip-estimate" in argv

    def test_passes_a_token_when_there_is_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _returns(monkeypatch, "{}", 0)
        assert_engine_can_load(_REPO, _FILE, "hf_secret")
        argv = calls[-1]
        assert argv[argv.index("--token") + 1] == "hf_secret"

    def test_omits_the_token_flag_when_there_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A blank token would be sent as an empty bearer and rejected."""
        calls = _returns(monkeypatch, "{}", 0)
        assert_engine_can_load(_REPO, _FILE, "")
        assert "--token" not in calls[-1]


class TestNamedQuant:
    def test_reads_the_type_out_of_the_message(self) -> None:
        assert _named_quant(_TENSOR_FAILURE) == "GGMLType(42)"

    def test_keeps_the_whole_message_when_no_type_is_named(self) -> None:
        assert _named_quant("read tensor info 3: something new") == (
            "read tensor info 3: something new"
        )
