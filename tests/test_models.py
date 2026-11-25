"""Tests for models.py: RAM detection, model selection, picker UI, auto-install."""

import io
from unittest import mock

import pytest
from rich.progress import Progress

from conftest import PICKS_CHAT, install_fake_model
from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.modelhub import models
from lilbee.modelhub.models import ModelInfo
from tests._mock_effects import repeat_last
from tests._unreadable import POSIX_DENIES_READS


class TestModelCatalog:
    def test_not_empty(self):
        assert len(models._get_model_catalog(hf_client=get_services().hf_client)) > 0

    def test_all_model_info(self):
        for m in models._get_model_catalog(hf_client=get_services().hf_client):
            assert isinstance(m, ModelInfo)

    def test_derived_from_catalog(self):
        """The model catalog entries match catalog.py's PICKS_CHAT."""

        assert len(models._get_model_catalog(hf_client=get_services().hf_client)) == len(PICKS_CHAT)
        for mc, fc in zip(
            models._get_model_catalog(hf_client=get_services().hf_client), PICKS_CHAT, strict=True
        ):
            assert mc.ref == fc.ref

    def test_frozen(self):
        with pytest.raises(AttributeError):
            models._get_model_catalog(hf_client=get_services().hf_client)[0].ref = "nope"  # type: ignore[misc]


class TestGetSystemRamGb:
    def test_unix_ram_detection(self):
        mock_sysconf = mock.Mock(
            side_effect=lambda key: {
                "SC_PHYS_PAGES": 4194304,
                "SC_PAGE_SIZE": 4096,
            }[key]
        )
        with (
            mock.patch.object(models.sys, "platform", "linux"),
            mock.patch("os.sysconf", mock_sysconf, create=True),
        ):
            ram = models.get_system_ram_gb()
            assert abs(ram - 16.0) < 0.01

    def test_fallback_on_error(self):
        with (
            mock.patch.object(models.sys, "platform", "linux"),
            mock.patch("os.sysconf", side_effect=OSError("not supported"), create=True),
        ):
            assert models.get_system_ram_gb() == 8.0

    def test_windows_ram_detection(self):
        """Mock ctypes.windll to simulate Windows RAM detection."""
        mock_windll = mock.MagicMock()

        def fake_global_memory(byref_stat):
            stat = byref_stat._obj
            stat.ullTotalPhys = 16 * 1024**3

        mock_windll.kernel32.GlobalMemoryStatusEx.side_effect = fake_global_memory

        with (
            mock.patch.object(models.sys, "platform", "win32"),
            mock.patch("ctypes.windll", mock_windll, create=True),
        ):
            ram = models.get_system_ram_gb()
            assert abs(ram - 16.0) < 0.01

    def test_windows_fallback_on_error(self):
        mock_windll = mock.MagicMock()
        mock_windll.kernel32.GlobalMemoryStatusEx.side_effect = OSError("fail")
        with (
            mock.patch.object(models.sys, "platform", "win32"),
            mock.patch("ctypes.windll", mock_windll, create=True),
        ):
            assert models.get_system_ram_gb() == 8.0


class TestGetFreeDiskGb:
    def test_returns_free_space(self, tmp_path):
        usage = mock.Mock(free=50 * 1024**3)
        with mock.patch("shutil.disk_usage", return_value=usage):
            assert models.get_free_disk_gb(tmp_path) == 50.0

    def test_walks_up_to_existing_parent(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        usage = mock.Mock(free=10 * 1024**3)
        with mock.patch("shutil.disk_usage", return_value=usage) as mock_du:
            result = models.get_free_disk_gb(deep)
            assert result == 10.0
            mock_du.assert_called_once_with(tmp_path)


class TestPickDefaultModel:
    def test_returns_model_info(self):
        result = models.pick_default_model(8.0, hf_client=get_services().hf_client)
        assert isinstance(result, ModelInfo)

    def test_low_ram_picks_small(self):
        result = models.pick_default_model(4.0, hf_client=get_services().hf_client)
        assert result.min_ram_gb <= 4.0

    def test_8gb_ram(self):
        result = models.pick_default_model(8.0, hf_client=get_services().hf_client)
        assert result.min_ram_gb <= 8.0

    def test_16gb_ram(self):
        result = models.pick_default_model(16.0, hf_client=get_services().hf_client)
        assert result.min_ram_gb <= 16.0

    def test_32gb_ram(self):
        result = models.pick_default_model(32.0, hf_client=get_services().hf_client)
        assert result.min_ram_gb <= 32.0

    def test_raises_when_the_catalog_is_empty(self, monkeypatch):
        """HuggingFace unreachable must surface as a message, not an IndexError."""
        monkeypatch.setattr(models, "_get_model_catalog", lambda **_: ())
        with pytest.raises(RuntimeError, match="Could not reach HuggingFace"):
            models.pick_default_model(16.0, hf_client=get_services().hf_client)

    def test_tiny_ram_picks_smallest(self):
        """No repo id is stable any more, so assert the fit rule, not a name."""
        result = models.pick_default_model(2.0, hf_client=get_services().hf_client)
        assert result.min_ram_gb <= 2.0
        fitting = [
            m
            for m in models._get_model_catalog(hf_client=get_services().hf_client)
            if m.min_ram_gb <= 2.0
        ]
        assert result.size_gb == max(m.size_gb for m in fitting)


class TestModelDownloadSizeGb:
    def test_known_models(self):
        first = models._get_model_catalog(hf_client=get_services().hf_client)[0]
        assert (
            models._model_download_size_gb(first.ref, hf_client=get_services().hf_client)
            == first.size_gb
        )

    def test_unknown_model_returns_fallback(self):
        result = models._model_download_size_gb(
            "unknown:latest", hf_client=get_services().hf_client
        )
        assert isinstance(result, float)
        assert result > 0


class TestDisplayModelPicker:
    def test_renders_table(self, capsys):
        recommended = models.display_model_picker(16.0, 50.0, hf_client=get_services().hf_client)
        captured = capsys.readouterr()
        assert "Available Models" in captured.err
        assert (
            models._get_model_catalog(hf_client=get_services().hf_client)[0].display_name
            in captured.err
        )
        assert isinstance(recommended, ModelInfo)

    def test_recommended_highlighted(self, capsys):
        recommended = models.display_model_picker(32.0, 100.0, hf_client=get_services().hf_client)
        assert recommended.min_ram_gb <= 32.0
        captured = capsys.readouterr()
        # The star marker should be in the output
        assert "\u2605" in captured.err

    def test_disk_warning_with_low_space(self, capsys):
        models.display_model_picker(32.0, 3.0, hf_client=get_services().hf_client)
        captured = capsys.readouterr()
        # Table still renders with disk info showing low space
        assert "3.0 GB free disk" in captured.err
        assert "Available Models" in captured.err

    def test_shows_system_stats(self, capsys):
        models.display_model_picker(16.0, 42.5, hf_client=get_services().hf_client)
        captured = capsys.readouterr()
        assert "16 GB RAM" in captured.err
        assert "42.5 GB free disk" in captured.err

    def test_shows_browse_link(self, capsys):
        models.display_model_picker(8.0, 50.0, hf_client=get_services().hf_client)
        captured = capsys.readouterr()
        assert models.MODELS_BROWSE_URL in captured.err


class TestPromptModelChoice:
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    def test_default_choice(self, mock_disk_estimate):
        with mock.patch("builtins.input", return_value=""):
            result = models.prompt_model_choice(8.0, hf_client=get_services().hf_client)
        assert isinstance(result, ModelInfo)
        # Default = recommended for 8 GB
        assert result == models.pick_default_model(8.0, hf_client=get_services().hf_client)

    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    def test_numeric_choice(self, mock_disk_estimate):
        with mock.patch("builtins.input", return_value="1"):
            result = models.prompt_model_choice(8.0, hf_client=get_services().hf_client)
        assert result == models._get_model_catalog(hf_client=get_services().hf_client)[0]

    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    def test_invalid_then_valid(self, mock_disk_estimate):
        with mock.patch("builtins.input", side_effect=repeat_last("abc", "99", "2")):
            result = models.prompt_model_choice(8.0, hf_client=get_services().hf_client)
        assert result == models._get_model_catalog(hf_client=get_services().hf_client)[1]

    @pytest.mark.parametrize("signal", [EOFError, KeyboardInterrupt])
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    def test_end_of_input_is_not_a_choice(self, mock_disk_estimate, signal):
        with (
            mock.patch("builtins.input", side_effect=signal),
            pytest.raises(RuntimeError, match="lilbee model pull"),
        ):
            models.prompt_model_choice(8.0, hf_client=get_services().hf_client)


class TestValidateDiskAndPull:
    @mock.patch.object(models, "pull_with_progress")
    def test_pulls_and_returns_ref(self, mock_pull):
        ref = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
        info = ModelInfo(
            ref=ref,
            display_name="Qwen3 0.6B",
            size_gb=1.0,
            min_ram_gb=4,
            description="test",
        )
        result = models.validate_disk_and_pull(info, 50.0)
        mock_pull.assert_called_once_with(ref, console=None, report_no_engine=True)
        assert result == ref

    def test_insufficient_disk_raises(self):
        info = ModelInfo(
            ref="org/Big-GGUF/big-Q4_K_M.gguf",
            display_name="Big",
            size_gb=20.0,
            min_ram_gb=32,
            description="big",
        )
        with pytest.raises(RuntimeError, match="Not enough disk space"):
            models.validate_disk_and_pull(info, 5.0)


class TestPullWithProgress:
    def test_calls_manager_pull(self):
        from lilbee.app.services import set_services

        mock_manager = mock.MagicMock()

        def fake_pull(
            model, source, *, on_bytes=None, on_notice=None, on_retry=None, report_no_engine=True
        ):
            if on_bytes:
                on_bytes(100, 100)
            return

        mock_manager.pull.side_effect = fake_pull
        set_services(mock.MagicMock(model_manager=mock_manager))
        models.pull_with_progress("test-model")
        mock_manager.pull.assert_called_once()

    def test_handles_zero_total(self):
        from lilbee.app.services import set_services

        mock_manager = mock.MagicMock()

        def fake_pull(
            model, source, *, on_bytes=None, on_notice=None, on_retry=None, report_no_engine=True
        ):
            if on_bytes:
                on_bytes(0, 0)
            return

        mock_manager.pull.side_effect = fake_pull
        set_services(mock.MagicMock(model_manager=mock_manager))
        models.pull_with_progress("test-model")


class TestPullWithProgressProjector:
    """The first-run picker pull runs the shared projector step after the download."""

    @staticmethod
    def _services() -> mock.MagicMock:
        from lilbee.app.services import set_services

        services = mock.MagicMock(model_manager=mock.MagicMock())
        set_services(services)
        return services

    def test_fetches_projector_after_download(self):
        services = self._services()
        order: list[str] = []
        services.model_manager.pull.side_effect = lambda *a, **k: order.append("pull")
        with mock.patch(
            "lilbee.app.models.ensure_vision_projector",
            side_effect=lambda ref, **k: order.append(f"projector:{ref}:{k['warn_unreadable']}"),
        ):
            models.pull_with_progress("acme/vl-GGUF")
        assert order == ["pull", "projector:acme/vl-GGUF:True"]

    def test_prints_projector_warning(self):
        self._services()
        console = models.PlainConsole(file=io.StringIO(), width=200)
        with mock.patch(
            "lilbee.app.models.ensure_vision_projector",
            return_value="acme/vl is named as a vision model, but lists no projector file.",
        ):
            models.pull_with_progress("acme/vl-GGUF", console=console)
        out = console.file.getvalue()
        assert "Warning: acme/vl is named as a vision model" in out
        assert "Model 'acme/vl-GGUF' ready." in out

    def test_prints_the_size_notice_with_the_projector_warning(self):
        services = self._services()
        services.model_manager.pull.side_effect = lambda *a, on_notice=None, **k: on_notice(
            "The size of this model is unknown, so 8.0 GiB is reserved for it."
        )
        console = models.PlainConsole(file=io.StringIO(), width=300)
        with mock.patch("lilbee.app.models.ensure_vision_projector", return_value="no projector"):
            models.pull_with_progress("acme/vl-GGUF", console=console)
        out = console.file.getvalue()
        assert (
            "Warning: The size of this model is unknown, so 8.0 GiB is reserved for it. "
            "no projector" in out
        )

    def test_shows_a_failed_attempt_in_the_progress_line_until_bytes_arrive(self):
        services = self._services()

        def _pull(*_a, on_bytes, on_retry, **_k):
            on_retry(1, 6, "Network error downloading acme/vl-GGUF: timed out")
            on_bytes(10, 100)

        services.model_manager.pull.side_effect = _pull
        details: list[str] = []

        class _Recording(Progress):
            def update(self, task_id, **fields):
                if "detail" in fields:
                    details.append(fields["detail"])
                super().update(task_id, **fields)

        console = models.PlainConsole(file=io.StringIO(), width=300)
        with (
            mock.patch("lilbee.app.models.ensure_vision_projector", return_value=None),
            mock.patch.object(
                models, "live_progress", lambda *cols, **opts: _Recording(*cols, **opts)
            ),
        ):
            models.pull_with_progress("acme/vl-GGUF", console=console)
        notice = (
            "Network error downloading acme/vl-GGUF: timed out. "
            "Attempt 1 of 6 failed; trying again."
        )
        assert details == [notice, ""]
        assert notice not in console.file.getvalue()

    def test_prints_no_warning_when_nothing_is_wrong(self):
        self._services()
        console = models.PlainConsole(file=io.StringIO(), width=300)
        with mock.patch("lilbee.app.models.ensure_vision_projector", return_value=None):
            models.pull_with_progress("acme/vl-GGUF", console=console)
        assert "Warning" not in console.file.getvalue()

    def test_unfetchable_projector_is_an_error(self):
        self._services()
        with (
            mock.patch(
                "lilbee.app.models.ensure_vision_projector",
                side_effect=RuntimeError("Fetched the projector p.gguf for r, but not found"),
            ),
            pytest.raises(RuntimeError, match="Fetched the projector"),
        ):
            models.pull_with_progress("acme/vl-GGUF")


class TestEnsureChatModelAtEndOfInput:
    """A tty-looking stdin that is already at end of input pulls nothing."""

    @pytest.mark.parametrize("signal", [EOFError, KeyboardInterrupt])
    @mock.patch.object(models, "pull_with_progress")
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    @mock.patch.object(models, "get_system_ram_gb", return_value=32.0)
    @mock.patch.object(models, "list_installed_models", return_value=[])
    def test_refuses_and_pulls_nothing(self, _list, _ram, _disk, mock_pull, signal):
        with (
            mock.patch.object(models.sys.stdin, "isatty", return_value=True),
            mock.patch("builtins.input", side_effect=signal),
            pytest.raises(RuntimeError, match="lilbee model pull"),
        ):
            models.ensure_chat_model(hf_client=get_services().hf_client)
        mock_pull.assert_not_called()

    @mock.patch.object(models, "pull_with_progress")
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    @mock.patch.object(models, "get_system_ram_gb", return_value=32.0)
    @mock.patch.object(models, "list_installed_models", return_value=[])
    def test_enter_on_empty_line_pulls_recommended(self, _list, _ram, _disk, mock_pull):
        with (
            mock.patch.object(models.sys.stdin, "isatty", return_value=True),
            mock.patch("builtins.input", return_value=""),
        ):
            pulled = models.ensure_chat_model(hf_client=get_services().hf_client)
        assert pulled == models.pick_default_model(32.0, hf_client=get_services().hf_client).ref
        mock_pull.assert_called_once()


class TestEnsureChatModel:
    """ensure_chat_model bootstraps only when no chat-task model is installed.

    It consults the task-aware ``list_installed_models`` (chat-task only), so a
    pulled vision/reranker model -- or a remote non-chat model -- no longer
    short-circuits the bootstrap.
    """

    def test_noop_when_chat_model_exists(self):
        with mock.patch.object(models, "list_installed_models", return_value=["llama3:latest"]):
            assert models.ensure_chat_model(hf_client=get_services().hf_client) is None  # no pull

    @mock.patch.object(models, "pull_with_progress")
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    @mock.patch.object(models, "get_system_ram_gb", return_value=32.0)
    @mock.patch.object(models, "list_installed_models", return_value=[])
    def test_non_interactive_errors_instead_of_pulling(
        self, _mock_list, mock_vram_estimate, mock_disk_estimate, mock_pull
    ):
        """No terminal to prompt on means error with guidance, never a silent download."""
        with (
            mock.patch.object(models.sys.stdin, "isatty", return_value=False),
            pytest.raises(RuntimeError, match="lilbee model pull"),
        ):
            models.ensure_chat_model(hf_client=get_services().hf_client)
        mock_pull.assert_not_called()

    @mock.patch.object(models, "pull_with_progress")
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    @mock.patch.object(models, "get_system_ram_gb", return_value=16.0)
    @mock.patch.object(models, "list_installed_models", return_value=[])
    def test_interactive_uses_picker(
        self, _mock_list, mock_vram_estimate, mock_disk_estimate, mock_pull
    ):
        with (
            mock.patch.object(models.sys.stdin, "isatty", return_value=True),
            mock.patch("builtins.input", return_value="1"),
        ):
            models.ensure_chat_model(hf_client=get_services().hf_client)
        mock_pull.assert_called_once_with(
            models._get_model_catalog(hf_client=get_services().hf_client)[0].ref,
            console=None,
            report_no_engine=True,
        )

    @mock.patch.object(models, "get_free_disk_gb", return_value=0.01)
    @mock.patch.object(models, "get_system_ram_gb", return_value=32.0)
    @mock.patch.object(models, "list_installed_models", return_value=[])
    def test_insufficient_disk_raises(self, _mock_list, mock_vram_estimate, mock_disk_estimate):
        with (
            mock.patch.object(models.sys.stdin, "isatty", return_value=True),
            mock.patch("builtins.input", return_value="1"),
            pytest.raises(RuntimeError, match="Not enough disk space"),
        ):
            models.ensure_chat_model(hf_client=get_services().hf_client)

    @mock.patch.object(models, "pull_with_progress")
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    @mock.patch.object(models, "get_system_ram_gb", return_value=16.0)
    def test_non_chat_only_install_still_pulls(self, mock_ram, mock_disk, mock_pull):
        """Regression (bb-ziks.67): an installed reranker/vision model is not a chat
        model, so the real (task-filtered) list_installed_models excludes it and the
        bootstrap still engages the picker rather than short-circuiting. Drives the
        real list_installed_models with a real ModelManifest, which demotes the
        reranker by name when it is built."""
        from lilbee.catalog.types import ModelTask
        from lilbee.modelhub.registry import ModelManifest

        # A manifest that declares task="chat" but whose ref names a reranker; the
        # manifest demotes it by name so it never counts as a chat model.
        reranker = ModelManifest(
            hf_repo="BAAI/bge-reranker-v2-m3",
            gguf_filename="bge-reranker-v2-m3-Q8_0.gguf",
            size_bytes=1,
            task=ModelTask.CHAT,
            downloaded_at="2026-05-15T00:00:00+00:00",
        )
        with (
            mock.patch.object(models, "ModelRegistry") as mock_registry,
            mock.patch("lilbee.modelhub.model_manager.classify_all_remote_models", return_value=[]),
            mock.patch.object(models.sys.stdin, "isatty", return_value=True),
            mock.patch("builtins.input", return_value=""),
        ):
            mock_registry.return_value.list_installed.return_value = [reranker]
            # The reranker is excluded, so the chat-task list is empty...
            assert models.list_installed_models() == []
            # ...and the bootstrap therefore prompts for a real chat model.
            pulled = models.ensure_chat_model(hf_client=get_services().hf_client)
        assert pulled == models.pick_default_model(16.0, hf_client=get_services().hf_client).ref
        mock_pull.assert_called_once()


# ensure_tag was removed alongside the alias system. Tag normalisation has
# no meaning under the new HF-keyed identity, so callers either pass a
# canonical ref through directly or fail loudly via parse_model_ref.


class TestPickDefaultModelFallback:
    def test_nothing_fits_falls_back_to_the_smallest(self, monkeypatch):
        """A host too small for every pick still gets an honest offer."""
        catalog = tuple(models._get_model_catalog(hf_client=get_services().hf_client))
        monkeypatch.setattr(models, "_get_model_catalog", lambda **_: catalog)
        result = models.pick_default_model(0.5, hf_client=get_services().hf_client)
        assert result.size_gb == min(m.size_gb for m in catalog)


@POSIX_DENIES_READS
class TestUnreadableRegistry:
    """A registry that cannot be read is unknown, not an empty install set."""

    def test_list_installed_models_surfaces_an_unreadable_registry(self) -> None:
        from lilbee.modelhub.registry import repo_to_dir
        from tests._unreadable import unreadable

        repo = "Qwen/Qwen3-0.6B-GGUF"
        install_fake_model(repo, "Qwen3-0.6B-Q4_K_M.gguf", "chat")
        repo_dir = cfg.models_dir / "manifests" / repo_to_dir(repo)

        with (
            mock.patch("lilbee.modelhub.model_manager.classify_all_remote_models", return_value=[]),
            unreadable(repo_dir),
            pytest.raises(OSError),
        ):
            models.list_installed_models()


class TestMalformedServerListing:
    """An unusable remote server is a different event from an unreadable registry."""

    def test_list_installed_models_keeps_the_native_half(self) -> None:
        """A listing neither strategy can walk drops its own names and nothing else."""
        ref = install_fake_model("acme/Jan-v3.5-4B-GGUF", "jan.gguf", "chat")
        response = mock.Mock()
        response.raise_for_status = mock.Mock()
        # Both parse loops run outside the request guard, so a null listing
        # raises past the strategy rather than yielding an empty list.
        response.json.return_value = {"models": None, "data": None}

        with mock.patch("lilbee.modelhub.model_manager.discovery._http_get", return_value=response):
            assert models.list_installed_models() == [ref]


class TestEnsureChatModelWithNoEngine:
    """A chat model is installed but no installed engine loads it: say so, name the fix."""

    @pytest.fixture
    def no_engine(self, monkeypatch):
        monkeypatch.setattr("lilbee.providers.fleet.engines.choice.installed_engines", dict)
        monkeypatch.setattr("lilbee.modelhub.models.installed_engines", dict)
        monkeypatch.setattr("lilbee.modelhub.model_manager.classify_all_remote_models", lambda: [])

    @pytest.mark.parametrize("interactive", [False, True])
    def test_names_the_missing_engine_and_the_command(self, no_engine, interactive):
        ref = install_fake_model("tiny/Tiny-1B-GGUF", "tiny-1b-Q4_K_M.gguf", "chat")
        with (
            mock.patch.object(models.sys.stdin, "isatty", return_value=interactive),
            mock.patch.object(models, "pull_with_progress") as mock_pull,
            pytest.raises(RuntimeError, match="No installed engine loads the gguf format") as err,
        ):
            models.ensure_chat_model(hf_client=get_services().hf_client)
        assert ref in str(err.value)
        assert "lilbee engine list" in str(err.value)
        assert "No chat model is installed" not in str(err.value)
        mock_pull.assert_not_called()

    def test_with_no_model_installed_the_pull_guidance_stays(self, no_engine):
        with (
            mock.patch.object(models.sys.stdin, "isatty", return_value=False),
            pytest.raises(RuntimeError, match="No chat model is installed"),
        ):
            models.ensure_chat_model(hf_client=get_services().hf_client)
