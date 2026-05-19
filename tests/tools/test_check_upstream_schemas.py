"""Tests for the upstream-schema retirement watcher."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from tools.check_upstream_schemas import (
    FamilyCheck,
    RetirementStatus,
    check_drift,
    check_family,
    list_local_schemas,
    load_upstream_repos,
    main,
    render_report,
    run,
)


def _write_repos(path: Path, mapping: dict[str, str]) -> None:
    payload = {family: {"repo": repo_id} for family, repo_id in mapping.items()}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_schema(dir_path: Path, family: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{family}.json").write_text("{}", encoding="utf-8")


def _stub_tokenizer_config(tmp_path: Path, **fields: Any) -> Path:
    config_path = tmp_path / "tokenizer_config.json"
    config_path.write_text(json.dumps(fields), encoding="utf-8")
    return config_path


class TestLoadUpstreamRepos:
    def test_returns_family_to_repo_map(self, tmp_path: Path) -> None:
        path = tmp_path / "repos.json"
        _write_repos(path, {"qwen3": "Qwen/Qwen3-8B"})
        assert load_upstream_repos(path) == {"qwen3": "Qwen/Qwen3-8B"}


class TestListLocalSchemas:
    def test_skips_underscore_prefixed_files(self, tmp_path: Path) -> None:
        _make_schema(tmp_path, "qwen3")
        _make_schema(tmp_path, "mistral")
        (tmp_path / "_upstream_repos.json").write_text("{}", encoding="utf-8")
        assert list_local_schemas(tmp_path) == {"qwen3", "mistral"}


class TestCheckDrift:
    def test_detects_schema_without_repo_entry(self, tmp_path: Path) -> None:
        _make_schema(tmp_path, "qwen3")
        _make_schema(tmp_path, "mistral")
        missing_repos, missing_schemas = check_drift(tmp_path, {"qwen3": "Qwen/Qwen3-8B"})
        assert missing_repos == {"mistral"}
        assert missing_schemas == set()

    def test_detects_repo_entry_without_schema(self, tmp_path: Path) -> None:
        _make_schema(tmp_path, "qwen3")
        missing_repos, missing_schemas = check_drift(
            tmp_path,
            {"qwen3": "Qwen/Qwen3-8B", "phantom": "fake/Phantom"},
        )
        assert missing_repos == set()
        assert missing_schemas == {"phantom"}


class TestCheckFamily:
    def test_ready_when_response_schema_populated(self, tmp_path: Path, monkeypatch) -> None:
        config_path = _stub_tokenizer_config(tmp_path, response_schema={"type": "object"})
        monkeypatch.setattr(
            "tools.check_upstream_schemas.hf_hub_download",
            lambda **_kw: str(config_path),
        )
        check = check_family("qwen3", "Qwen/Qwen3-8B")
        assert check == FamilyCheck("qwen3", "Qwen/Qwen3-8B", RetirementStatus.READY)

    def test_pending_when_response_schema_absent(self, tmp_path: Path, monkeypatch) -> None:
        config_path = _stub_tokenizer_config(tmp_path, model_type="qwen3")
        monkeypatch.setattr(
            "tools.check_upstream_schemas.hf_hub_download",
            lambda **_kw: str(config_path),
        )
        check = check_family("qwen3", "Qwen/Qwen3-8B")
        assert check.status is RetirementStatus.PENDING

    def test_pending_when_response_schema_empty(self, tmp_path: Path, monkeypatch) -> None:
        config_path = _stub_tokenizer_config(tmp_path, response_schema={})
        monkeypatch.setattr(
            "tools.check_upstream_schemas.hf_hub_download",
            lambda **_kw: str(config_path),
        )
        check = check_family("qwen3", "Qwen/Qwen3-8B")
        assert check.status is RetirementStatus.PENDING

    def test_blocked_on_gated_repo(self, monkeypatch) -> None:
        response = httpx.Response(status_code=401, request=httpx.Request("GET", "/"))

        def _raise(**_kw: Any) -> str:
            raise GatedRepoError("gated", response=response)

        monkeypatch.setattr("tools.check_upstream_schemas.hf_hub_download", _raise)
        check = check_family("cohere", "CohereForAI/c4ai-command-r-08-2024")
        assert check.status is RetirementStatus.BLOCKED
        assert "gated repo" in check.detail

    def test_blocked_on_repo_not_found(self, monkeypatch) -> None:
        response = httpx.Response(status_code=404, request=httpx.Request("GET", "/"))

        def _raise(**_kw: Any) -> str:
            raise RepositoryNotFoundError("gone", response=response)

        monkeypatch.setattr("tools.check_upstream_schemas.hf_hub_download", _raise)
        check = check_family("phantom", "fake/Phantom")
        assert check.status is RetirementStatus.BLOCKED
        assert "repo not found" in check.detail

    def test_blocked_on_network_error(self, monkeypatch) -> None:
        def _raise(**_kw: Any) -> str:
            raise OSError("dns timeout")

        monkeypatch.setattr("tools.check_upstream_schemas.hf_hub_download", _raise)
        check = check_family("qwen3", "Qwen/Qwen3-8B")
        assert check.status is RetirementStatus.BLOCKED
        assert "network/IO" in check.detail

    def test_blocked_on_parse_error(self, tmp_path: Path, monkeypatch) -> None:
        bad_path = tmp_path / "tokenizer_config.json"
        bad_path.write_text("not json", encoding="utf-8")
        monkeypatch.setattr(
            "tools.check_upstream_schemas.hf_hub_download",
            lambda **_kw: str(bad_path),
        )
        check = check_family("qwen3", "Qwen/Qwen3-8B")
        assert check.status is RetirementStatus.BLOCKED
        assert "parse" in check.detail

    def test_blocked_on_unexpected_exception(self, monkeypatch) -> None:
        """Unknown huggingface_hub error subclasses surface as BLOCKED, not a crash."""

        class _SomeFutureHfError(Exception):
            pass

        def _raise(**_kw: Any) -> str:
            raise _SomeFutureHfError("upstream broke a contract")

        monkeypatch.setattr("tools.check_upstream_schemas.hf_hub_download", _raise)
        check = check_family("qwen3", "Qwen/Qwen3-8B")
        assert check.status is RetirementStatus.BLOCKED
        assert "_SomeFutureHfError" in check.detail


class TestLoadUpstreamReposValidation:
    def test_rejects_non_object_root(self, tmp_path: Path) -> None:
        path = tmp_path / "repos.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="expected an object"):
            load_upstream_repos(path)

    def test_rejects_entry_missing_repo_key(self, tmp_path: Path) -> None:
        path = tmp_path / "repos.json"
        path.write_text(json.dumps({"qwen3": {"not_repo": "Qwen/Qwen3-8B"}}), encoding="utf-8")
        with pytest.raises(ValueError, match=r"qwen3.*repo"):
            load_upstream_repos(path)

    def test_rejects_entry_with_non_string_repo(self, tmp_path: Path) -> None:
        path = tmp_path / "repos.json"
        path.write_text(json.dumps({"qwen3": {"repo": 42}}), encoding="utf-8")
        with pytest.raises(ValueError, match=r"qwen3.*repo"):
            load_upstream_repos(path)


class TestRenderReport:
    def test_renders_each_status_section(self) -> None:
        checks = [
            FamilyCheck("qwen3", "Qwen/Qwen3-8B", RetirementStatus.READY),
            FamilyCheck("mistral", "mistralai/Mistral-7B-Instruct-v0.3", RetirementStatus.PENDING),
            FamilyCheck(
                "cohere",
                "CohereForAI/c4ai-command-r-08-2024",
                RetirementStatus.BLOCKED,
                "gated repo",
            ),
        ]
        report = render_report(checks)
        assert "## Ready to retire" in report
        assert "### `qwen3` migrated upstream" in report
        assert "## Pending upstream" in report
        assert "mistral" in report
        assert "## Could not check" in report
        assert "gated repo" in report

    def test_renders_empty_when_no_checks(self) -> None:
        report = render_report([])
        assert "No families to check" in report

    def test_ready_block_includes_retirement_checklist(self) -> None:
        checks = [FamilyCheck("qwen3", "Qwen/Qwen3-8B", RetirementStatus.READY)]
        report = render_report(checks)
        assert "Remove `src/lilbee/providers/worker/response_parser/schemas/qwen3.json`" in report
        assert "Remove `TemplateFamily.QWEN3`" in report
        assert "test_parse.py" in report
        assert "test_families.py" in report
        assert "_upstream_repos.json" in report


class TestRun:
    def test_raises_when_schema_lacks_repo_entry(self, tmp_path: Path) -> None:
        schemas = tmp_path / "schemas"
        _make_schema(schemas, "qwen3")
        _make_schema(schemas, "mistral")
        repos = tmp_path / "repos.json"
        _write_repos(repos, {"qwen3": "Qwen/Qwen3-8B"})
        with pytest.raises(ValueError, match="without an upstream repo entry"):
            run(schemas_dir=schemas, upstream_repos_file=repos)

    def test_raises_when_repo_entry_lacks_schema(self, tmp_path: Path) -> None:
        schemas = tmp_path / "schemas"
        _make_schema(schemas, "qwen3")
        repos = tmp_path / "repos.json"
        _write_repos(repos, {"qwen3": "Qwen/Qwen3-8B", "phantom": "fake/Phantom"})
        with pytest.raises(ValueError, match="no matching local schema"):
            run(schemas_dir=schemas, upstream_repos_file=repos)

    def test_produces_report_when_drift_clean(self, tmp_path: Path, monkeypatch) -> None:
        schemas = tmp_path / "schemas"
        _make_schema(schemas, "qwen3")
        repos = tmp_path / "repos.json"
        _write_repos(repos, {"qwen3": "Qwen/Qwen3-8B"})
        config_path = _stub_tokenizer_config(tmp_path, response_schema={"type": "object"})
        monkeypatch.setattr(
            "tools.check_upstream_schemas.hf_hub_download",
            lambda **_kw: str(config_path),
        )
        report = run(schemas_dir=schemas, upstream_repos_file=repos)
        assert "Ready to retire" in report
        assert "qwen3" in report


class TestMain:
    def test_writes_report_to_stdout(self, tmp_path: Path, capsys, monkeypatch) -> None:
        schemas = tmp_path / "schemas"
        _make_schema(schemas, "qwen3")
        repos = tmp_path / "repos.json"
        _write_repos(repos, {"qwen3": "Qwen/Qwen3-8B"})
        config_path = _stub_tokenizer_config(tmp_path, model_type="qwen3")
        monkeypatch.setattr(
            "tools.check_upstream_schemas.hf_hub_download",
            lambda **_kw: str(config_path),
        )
        rc = main(
            [
                "--schemas-dir",
                str(schemas),
                "--upstream-repos-file",
                str(repos),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Pending upstream" in out
        assert "qwen3" in out


class TestRetirementStatus:
    def test_is_str_enum(self) -> None:
        assert RetirementStatus.READY.value == "ready"
        assert RetirementStatus.PENDING.value == "pending"
        assert RetirementStatus.BLOCKED.value == "blocked"
