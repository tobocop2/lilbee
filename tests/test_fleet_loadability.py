"""The gguf-parser probe that refuses a GGUF the bundled engine cannot decode."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lilbee.catalog.compat import UnsupportedQuantError
from lilbee.providers.base import ProviderError
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


def _returns_each(monkeypatch: pytest.MonkeyPatch, *results: tuple[str, int]) -> list[list[str]]:
    """Stub run_bounded to answer differently on each successive call."""
    calls: list[list[str]] = []
    answers = list(results)

    def _run(argv: list[str], **_kwargs: object) -> tuple[str, int]:
        calls.append(argv)
        return answers[min(len(calls) - 1, len(answers) - 1)]

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

    def test_a_refusal_that_names_no_type_is_still_a_verdict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The engine refusing on tensor size says as much as it refusing on type."""
        _returns(
            monkeypatch,
            "failed to parse GGUF file: tensor data needs 484218688 bytes but only "
            "457345184 follow the header: a tensor type's block layout does not match",
            1,
        )
        with pytest.raises(UnsupportedQuantError):
            assert_engine_can_load(_REPO, _FILE)

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

    def test_falls_back_to_the_reason_when_no_type_is_named(self) -> None:
        """The fit failure names no type, so the message carries the reason."""
        assert (
            _named_quant(
                "failed to parse GGUF file: tensor data needs 9 bytes but only 8 follow the header"
            )
            == "tensor data needs 9 bytes but only 8 follow the header"
        )

    def test_keeps_the_whole_message_when_it_has_no_prefix(self) -> None:
        assert _named_quant("something new") == "something new"


class TestTransientFailures:
    """A verdict is a property of the file, so it has to repeat."""

    def test_a_failure_that_does_not_repeat_is_not_a_verdict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A header read cut short reports as a parse failure and then succeeds."""
        truncated = (
            "failed to parse GGUF file: read metadata kv 27: read tokenizer.ggml.merges "
            "value: seek array[string] 138277: unexpected EOF"
        )
        calls = _returns_each(monkeypatch, (truncated, 1), ("{}", 0))
        assert assert_engine_can_load(_REPO, _FILE) is None
        assert len(calls) == 2

    def test_a_failure_that_repeats_is_a_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _returns(monkeypatch, _TENSOR_FAILURE, 1)
        with pytest.raises(UnsupportedQuantError):
            assert_engine_can_load(_REPO, _FILE)
        assert len(calls) == 2

    def test_a_parser_that_stops_running_is_not_a_verdict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The retry itself may fail to spawn; that is still not a refusal."""
        calls: list[list[str]] = []

        def _run(argv: list[str], **_kwargs: object) -> tuple[str, int]:
            calls.append(argv)
            if len(calls) == 1:
                return _TENSOR_FAILURE, 1
            raise OSError("no such binary")

        monkeypatch.setattr(loadability, "run_bounded", _run)
        assert assert_engine_can_load(_REPO, _FILE) is None


class TestNoParserAvailable:
    """An install with no engine yet has no parser to ask."""

    def test_a_missing_parser_is_not_a_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _unavailable() -> Path:
            raise ProviderError("gguf-parser binary not found.")

        monkeypatch.setattr(loadability, "resolve_gguf_parser", _unavailable)
        monkeypatch.setattr(
            loadability,
            "run_bounded",
            lambda *_a, **_kw: pytest.fail("ran a parser it could not resolve"),
        )
        assert assert_engine_can_load(_REPO, _FILE) is None


_GATED = "failed to parse GGUF file: open http file: stat: do head request: status code 401"

_FAKE_PARSER = """#!{python}
import json
import sys
from pathlib import Path

me = Path(sys.argv[0])
plan = json.loads(me.with_name(me.name + ".json").read_text())
calls = me.with_name(me.name + ".calls")
with calls.open("a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
n = len(calls.read_text().splitlines())
if any(item not in sys.argv for item in plan["needs"]):
    print(plan["refusal"])
    sys.exit(1)
out, code = plan["results"][min(n, len(plan["results"])) - 1]
print(out)
sys.exit(code)
"""


def _plant_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *results: tuple[str, int],
    needs: tuple[str, ...] = (),
) -> Path:
    """A stand-in parser run as a real process; a run missing any of *needs* gets a 401."""
    exe = tmp_path / "fake-gguf-parser"
    exe.write_text(_FAKE_PARSER.format(python=sys.executable), encoding="utf-8")
    exe.chmod(0o755)
    exe.with_name(exe.name + ".json").write_text(
        json.dumps({"results": results, "needs": needs, "refusal": _GATED}),
        encoding="utf-8",
    )
    monkeypatch.setattr(loadability, "resolve_gguf_parser", lambda: exe)
    return exe


def _recorded_runs(exe: Path) -> list[list[str]]:
    """The argv, minus the program, of every run the stand-in parser saw."""
    calls = exe.with_name(exe.name + ".calls")
    return [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in parser is a shebang script")
class TestAgainstARealParserProcess:
    """The probe starts a process; these checks see what that process was given and did."""

    def test_a_gated_repo_is_checked_when_the_token_is_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The token rides along with the repo and file; it does not replace them."""
        _plant_parser(
            tmp_path,
            monkeypatch,
            (_TENSOR_FAILURE, 1),
            needs=("--hf-repo", _REPO, "--hf-file", _FILE, "--token", "hf_secret"),
        )
        with pytest.raises(UnsupportedQuantError):
            assert_engine_can_load(_REPO, _FILE, "hf_secret")

    def test_a_gated_repo_is_not_a_verdict_without_the_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control: the stand-in does refuse a caller that sends no token."""
        _plant_parser(
            tmp_path,
            monkeypatch,
            (_TENSOR_FAILURE, 1),
            needs=("--hf-repo", _REPO, "--hf-file", _FILE, "--token", "hf_secret"),
        )
        assert assert_engine_can_load(_REPO, _FILE) is None

    def test_a_refusal_comes_from_running_the_resolved_parser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The program the probe starts is the one that was resolved."""
        exe = _plant_parser(tmp_path, monkeypatch, (_TENSOR_FAILURE, 1))
        with pytest.raises(UnsupportedQuantError) as excinfo:
            assert_engine_can_load(_REPO, _FILE)
        assert excinfo.value.quant == "GGMLType(42)"
        assert len(_recorded_runs(exe)) == 2

    def test_a_failure_that_clears_on_the_second_run_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second run is the same call as the first, so a read cut short recovers."""
        truncated = "failed to parse GGUF file: read metadata kv 27: unexpected EOF"
        exe = _plant_parser(tmp_path, monkeypatch, (truncated, 1), ("{}", 0))
        assert assert_engine_can_load(_REPO, _FILE, "hf_secret") is None
        first, second = _recorded_runs(exe)
        assert first == second
        assert first[first.index("--hf-repo") + 1] == _REPO
        assert first[first.index("--token") + 1] == "hf_secret"
