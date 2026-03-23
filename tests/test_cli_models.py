"""Tests for the ``lilbee models`` CLI command group."""

from __future__ import annotations

from unittest import mock

from typer.testing import CliRunner

from lilbee.cli.app import app

runner = CliRunner()


def _mock_manager(
    installed: list[str] | None = None,
    source_map: dict[str, str] | None = None,
) -> mock.MagicMock:
    """Create a mock ModelManager with common defaults."""
    from lilbee.model_manager import ModelSource

    manager = mock.MagicMock()
    manager.list_installed.return_value = installed or []

    source_map = source_map or {}
    source_enum_map = {name: ModelSource(src) for name, src in source_map.items()}
    manager.get_source.side_effect = lambda name: source_enum_map.get(name)
    manager.is_installed.side_effect = lambda name, source=None: name in (installed or [])
    return manager


class TestModelsList:
    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_empty_list(self, mock_get: mock.MagicMock) -> None:
        mock_get.return_value = _mock_manager()
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0
        assert "No models installed" in result.output

    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_shows_installed(self, mock_get: mock.MagicMock) -> None:
        mock_get.return_value = _mock_manager(
            installed=["qwen3:8b", "nomic.gguf"],
            source_map={"qwen3:8b": "ollama", "nomic.gguf": "native"},
        )
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0
        assert "qwen3:8b" in result.output
        assert "nomic.gguf" in result.output

    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_json_mode(self, mock_get: mock.MagicMock) -> None:
        mock_get.return_value = _mock_manager(
            installed=["qwen3:8b"],
            source_map={"qwen3:8b": "ollama"},
        )
        result = runner.invoke(app, ["--json", "models", "list"])
        assert result.exit_code == 0
        assert '"models list"' in result.output
        assert "qwen3:8b" in result.output

    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_source_filter(self, mock_get: mock.MagicMock) -> None:
        from lilbee.model_manager import ModelSource

        manager = _mock_manager(installed=["qwen3:8b"])
        manager.list_installed.side_effect = lambda src=None: (
            ["qwen3:8b"] if src == ModelSource.OLLAMA else []
        )
        mock_get.return_value = manager
        result = runner.invoke(app, ["models", "list", "--source", "ollama"])
        assert result.exit_code == 0


class TestModelsBrowse:
    @mock.patch("lilbee.catalog.get_catalog")
    def test_shows_catalog(self, mock_catalog: mock.MagicMock) -> None:
        from lilbee.catalog import CatalogModel, CatalogResult

        mock_catalog.return_value = CatalogResult(
            total=1,
            limit=20,
            offset=0,
            models=[
                CatalogModel(
                    "TestModel",
                    "test/repo",
                    "*.gguf",
                    2.0,
                    4,
                    "A test model",
                    True,
                    100,
                    "chat",
                ),
            ],
        )
        result = runner.invoke(app, ["models", "browse"])
        assert result.exit_code == 0
        assert "TestModel" in result.output

    @mock.patch("lilbee.catalog.get_catalog")
    def test_empty_catalog(self, mock_catalog: mock.MagicMock) -> None:
        from lilbee.catalog import CatalogResult

        mock_catalog.return_value = CatalogResult(total=0, limit=20, offset=0, models=[])
        result = runner.invoke(app, ["models", "browse"])
        assert result.exit_code == 0
        assert "No models found" in result.output

    @mock.patch("lilbee.catalog.get_catalog")
    def test_task_filter(self, mock_catalog: mock.MagicMock) -> None:
        from lilbee.catalog import CatalogResult

        mock_catalog.return_value = CatalogResult(total=0, limit=20, offset=0, models=[])
        runner.invoke(app, ["models", "browse", "--task", "embedding"])
        mock_catalog.assert_called_once()
        assert mock_catalog.call_args.kwargs.get("task") == "embedding"

    @mock.patch("lilbee.catalog.get_catalog")
    def test_json_mode(self, mock_catalog: mock.MagicMock) -> None:
        from lilbee.catalog import CatalogModel, CatalogResult

        mock_catalog.return_value = CatalogResult(
            total=1,
            limit=20,
            offset=0,
            models=[
                CatalogModel(
                    "TestModel",
                    "test/repo",
                    "*.gguf",
                    2.0,
                    4,
                    "A test",
                    True,
                    100,
                    "chat",
                ),
            ],
        )
        result = runner.invoke(app, ["--json", "models", "browse"])
        assert result.exit_code == 0
        assert '"models browse"' in result.output
        assert "TestModel" in result.output

    @mock.patch("lilbee.catalog.get_catalog")
    def test_pagination_hint(self, mock_catalog: mock.MagicMock) -> None:
        from lilbee.catalog import CatalogModel, CatalogResult

        models = [
            CatalogModel(f"m{i}", "r/r", "*.gguf", 1.0, 2, "d", False, 0, "chat") for i in range(3)
        ]
        mock_catalog.return_value = CatalogResult(total=10, limit=3, offset=0, models=models)
        result = runner.invoke(app, ["models", "browse", "--limit", "3"])
        assert result.exit_code == 0
        assert "--offset" in result.output

    @mock.patch("lilbee.catalog.get_catalog")
    def test_search_filter(self, mock_catalog: mock.MagicMock) -> None:
        from lilbee.catalog import CatalogResult

        mock_catalog.return_value = CatalogResult(total=0, limit=20, offset=0, models=[])
        runner.invoke(app, ["models", "browse", "--search", "nomic"])
        assert mock_catalog.call_args.kwargs.get("search") == "nomic"

    @mock.patch("lilbee.catalog.get_catalog")
    def test_featured_flag(self, mock_catalog: mock.MagicMock) -> None:
        from lilbee.catalog import CatalogResult

        mock_catalog.return_value = CatalogResult(total=0, limit=20, offset=0, models=[])
        runner.invoke(app, ["models", "browse", "--featured"])
        assert mock_catalog.call_args.kwargs.get("featured") is True

    @mock.patch("lilbee.catalog.get_catalog")
    def test_task_title(self, mock_catalog: mock.MagicMock) -> None:
        from lilbee.catalog import CatalogModel, CatalogResult

        mock_catalog.return_value = CatalogResult(
            total=1,
            limit=20,
            offset=0,
            models=[CatalogModel("m", "r/r", "*.gguf", 1.0, 2, "d", True, 0, "embedding")],
        )
        result = runner.invoke(app, ["models", "browse", "--task", "embedding"])
        assert "Embedding Models" in result.output


class TestModelsInstall:
    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_already_installed(self, mock_get: mock.MagicMock) -> None:
        mock_get.return_value = _mock_manager(installed=["qwen3:8b"])
        result = runner.invoke(app, ["models", "install", "qwen3:8b"])
        assert result.exit_code == 0
        assert "already installed" in result.output

    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_already_installed_json(self, mock_get: mock.MagicMock) -> None:
        mock_get.return_value = _mock_manager(installed=["qwen3:8b"])
        result = runner.invoke(app, ["--json", "models", "install", "qwen3:8b"])
        assert result.exit_code == 0
        assert "already_installed" in result.output

    @mock.patch("lilbee.models.pull_with_progress")
    @mock.patch("lilbee.catalog.find_catalog_entry")
    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_catalog_install(
        self,
        mock_get: mock.MagicMock,
        mock_find: mock.MagicMock,
        mock_pull: mock.MagicMock,
    ) -> None:
        from lilbee.catalog import CatalogModel

        mock_get.return_value = _mock_manager()
        mock_find.return_value = CatalogModel(
            "Nomic Embed Text v1.5",
            "nomic-ai/repo",
            "nomic.gguf",
            0.3,
            2,
            "desc",
            True,
            0,
            "embedding",
        )
        result = runner.invoke(app, ["models", "install", "Nomic Embed Text v1.5"])
        assert result.exit_code == 0
        mock_pull.assert_called_once_with("Nomic Embed Text v1.5")

    @mock.patch("lilbee.models.pull_with_progress")
    @mock.patch("lilbee.catalog.find_catalog_entry")
    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_catalog_install_json(
        self,
        mock_get: mock.MagicMock,
        mock_find: mock.MagicMock,
        mock_pull: mock.MagicMock,
    ) -> None:
        from lilbee.catalog import CatalogModel

        mock_get.return_value = _mock_manager()
        mock_find.return_value = CatalogModel(
            "Nomic Embed Text v1.5",
            "nomic-ai/repo",
            "nomic.gguf",
            0.3,
            2,
            "desc",
            True,
            0,
            "embedding",
        )
        result = runner.invoke(app, ["--json", "models", "install", "Nomic Embed Text v1.5"])
        assert result.exit_code == 0
        assert '"native"' in result.output

    @mock.patch("lilbee.model_manager.get_model_manager")
    @mock.patch("lilbee.catalog.find_catalog_entry", return_value=None)
    def test_ollama_fallback_json(
        self, mock_find: mock.MagicMock, mock_get: mock.MagicMock
    ) -> None:
        from lilbee.model_manager import ModelSource

        manager = _mock_manager()
        mock_get.return_value = manager
        result = runner.invoke(app, ["--json", "models", "install", "nomic-embed-text"])
        assert result.exit_code == 0
        assert '"ollama"' in result.output
        manager.pull.assert_called_once_with("nomic-embed-text", ModelSource.OLLAMA)

    @mock.patch("lilbee.model_manager.get_model_manager")
    @mock.patch("lilbee.catalog.find_catalog_entry", return_value=None)
    def test_ollama_fallback(self, mock_find: mock.MagicMock, mock_get: mock.MagicMock) -> None:
        from lilbee.model_manager import ModelSource

        manager = _mock_manager()
        mock_get.return_value = manager
        result = runner.invoke(app, ["models", "install", "nomic-embed-text"])
        assert result.exit_code == 0
        manager.pull.assert_called_once_with("nomic-embed-text", ModelSource.OLLAMA)

    @mock.patch("lilbee.model_manager.get_model_manager")
    @mock.patch("lilbee.catalog.find_catalog_entry", return_value=None)
    def test_ollama_failure(self, mock_find: mock.MagicMock, mock_get: mock.MagicMock) -> None:
        manager = _mock_manager()
        manager.pull.side_effect = RuntimeError("Connection refused")
        mock_get.return_value = manager
        result = runner.invoke(app, ["models", "install", "bad-model"])
        assert result.exit_code == 1
        assert "Connection refused" in result.output

    @mock.patch("lilbee.model_manager.get_model_manager")
    @mock.patch("lilbee.catalog.find_catalog_entry", return_value=None)
    def test_ollama_failure_json(self, mock_find: mock.MagicMock, mock_get: mock.MagicMock) -> None:
        manager = _mock_manager()
        manager.pull.side_effect = RuntimeError("Connection refused")
        mock_get.return_value = manager
        result = runner.invoke(app, ["--json", "models", "install", "bad-model"])
        assert result.exit_code == 1
        assert "Connection refused" in result.output


class TestModelsRemove:
    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_removes_model(self, mock_get: mock.MagicMock) -> None:
        manager = mock.MagicMock()
        manager.remove.return_value = True
        mock_get.return_value = manager
        result = runner.invoke(app, ["models", "remove", "qwen3:8b"])
        assert result.exit_code == 0
        assert "Removed" in result.output

    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_not_found(self, mock_get: mock.MagicMock) -> None:
        manager = mock.MagicMock()
        manager.remove.return_value = False
        mock_get.return_value = manager
        result = runner.invoke(app, ["models", "remove", "nonexistent"])
        assert result.exit_code == 1
        assert "Not found" in result.output

    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_json_mode(self, mock_get: mock.MagicMock) -> None:
        manager = mock.MagicMock()
        manager.remove.return_value = True
        mock_get.return_value = manager
        result = runner.invoke(app, ["--json", "models", "remove", "qwen3:8b"])
        assert result.exit_code == 0
        assert '"deleted": true' in result.output

    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_source_filter(self, mock_get: mock.MagicMock) -> None:
        from lilbee.model_manager import ModelSource

        manager = mock.MagicMock()
        manager.remove.return_value = True
        mock_get.return_value = manager
        result = runner.invoke(app, ["models", "remove", "qwen3:8b", "--source", "ollama"])
        assert result.exit_code == 0
        manager.remove.assert_called_once_with("qwen3:8b", ModelSource.OLLAMA)
