"""A transfer retries the faults another attempt can clear, and nothing else.

A dropped connection and a transient I/O error are worth another attempt; a
gated repo, a missing file and a cancellation are not. A transfer that goes
quiet raises nothing at all, so the parent process ends it instead. These
tests pin which faults retry, which are reported at once, and that every
download entry point runs under the retry loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.catalog import download as dl
from lilbee.catalog.models import CatalogModel
from lilbee.runtime.cancellation import TaskCancelledError


def _entry() -> CatalogModel:
    return CatalogModel(
        hf_repo="user/repo",
        gguf_filename="f.gguf",
        size_gb=1.0,
        min_ram_gb=2,
        description="d",
        featured=False,
        downloads=0,
        task="chat",
    )


def _config() -> dl.DownloadConfig:
    return dl.DownloadConfig(repo_id="user/repo", filename="f.gguf", token=None)


class TestTransientRetries:
    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        outcomes: list[Exception | Path],
    ) -> tuple[Path | None, int]:
        calls = {"n": 0}

        def _fake_transfer(entry: CatalogModel, config: dl.DownloadConfig) -> Path:
            outcome = outcomes[calls["n"]]
            calls["n"] += 1
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(dl, "_hf_download_or_translate", _fake_transfer)
        monkeypatch.setattr(dl.time, "sleep", lambda seconds: None)
        return dl._download_with_retry(_entry(), _config()), calls["n"]

    def test_a_network_fault_retries_and_finishes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        path = Path("/models/f.gguf")
        result, attempts = self._run(
            monkeypatch,
            [
                dl._TransientDownloadError("Network error downloading user/repo: reset"),
                dl._TransientDownloadError("I/O error downloading user/repo: closed"),
                path,
            ],
        )
        assert result == path
        assert attempts == 3

    def test_a_persistent_network_fault_fails_with_the_last_fault(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        faults = [
            dl._TransientDownloadError(f"Network error downloading user/repo: {reason}")
            for reason in "abc"
        ]
        with pytest.raises(RuntimeError, match=r"Network error downloading.*failed 3 times"):
            self._run(monkeypatch, faults)

    def test_the_backoff_grows_between_attempts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retrying a dropped connection at once repeats the drop."""
        slept: list[float] = []
        monkeypatch.setattr(dl.time, "sleep", lambda seconds: slept.append(seconds))
        monkeypatch.setattr(
            dl,
            "_hf_download_or_translate",
            lambda entry, config: (_ for _ in ()).throw(dl._TransientDownloadError("reset")),
        )

        with pytest.raises(RuntimeError, match="failed 3 times"):
            dl._download_with_retry(_entry(), _config())

        assert slept == [dl._RETRY_BACKOFF_SECONDS, 2 * dl._RETRY_BACKOFF_SECONDS]

    def test_a_real_error_propagates_without_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(RuntimeError, match="repo gone"):
            self._run(monkeypatch, [RuntimeError("repo gone")])

    def test_cancellation_is_never_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(TaskCancelledError):
            self._run(monkeypatch, [TaskCancelledError()])

    def test_an_assertion_failure_is_never_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A defect in the transfer is reported on the first attempt, never rerun."""
        with pytest.raises(AssertionError, match="checksum"):
            self._run(monkeypatch, [AssertionError("checksum mismatch")])

    def test_a_gated_repo_is_never_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A configuration error is not transient, so it is reported at once."""
        with pytest.raises(PermissionError, match="authentication"):
            self._run(monkeypatch, [PermissionError("needs authentication")])


class TestRetryIsOnEveryTransfer:
    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda entry: dl.download_model(entry), id="model"),
            pytest.param(lambda entry: dl.download_mmproj(entry), id="mmproj"),
        ],
    )
    def test_the_transfer_runs_under_the_retry_loop(
        self, call, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Both download entry points must go through the retry loop."""
        retried: list[str] = []
        gguf = tmp_path / "f.gguf"

        def _fake_retried(entry: CatalogModel, config: dl.DownloadConfig) -> Path:
            retried.append(config.filename)
            gguf.write_bytes(b"g")  # lands only when the transfer ran
            return gguf

        monkeypatch.setattr(dl, "_download_with_retry", _fake_retried)
        monkeypatch.setattr(dl, "_models_dir", lambda: tmp_path)
        monkeypatch.setattr(dl, "resolve_filename", lambda entry: "f.gguf")
        monkeypatch.setattr(dl, "fetch_remote_file", lambda *a: dl.RemoteFile(size=1, blob="abc"))
        monkeypatch.setattr(dl, "_resolve_mmproj_filename", lambda *a: "mmproj.gguf")
        monkeypatch.setattr(dl, "repo_has_mmproj", lambda repo: False)

        call(_entry())

        assert retried, "the transfer bypassed the retry loop"


class TestTransientErrorTranslation:
    """The hub's transient faults must reach the retry loop as transient."""

    def _download(self, monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
        def _raise(**_kwargs: object) -> str:
            raise error

        monkeypatch.setattr("huggingface_hub.hf_hub_download", _raise)
        dl._hf_download_or_translate(_entry(), _config())

    def test_a_network_timeout_is_transient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        with pytest.raises(dl._TransientDownloadError, match="Network error"):
            self._download(monkeypatch, httpx.ConnectError("refused"))

    def test_an_io_error_is_transient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(dl._TransientDownloadError, match="I/O error"):
            self._download(monkeypatch, OSError("disk hiccup"))
