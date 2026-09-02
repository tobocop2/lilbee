"""The generated settings reference must keep agreeing with the real surfaces.

``docs/settings.md`` tells a caller whether MCP, HTTP, the TUI or the CLI can
change a given setting, and whether the change needs a reconnect or a reindex.
Those claims are only useful while they match the code, so they are asserted
here rather than trusted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from lilbee.config_meta import PUBLIC_CONFIG_FIELDS, WRITABLE_CONFIG_FIELDS
from lilbee.core.config import Config, cfg
from lilbee.mcp_server import TOOL_GATE_SETTINGS, build_mcp_server
from lilbee.providers.roles import MODEL_ROLE_FIELDS

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR = REPO_ROOT / "tools" / "gen_settings_reference.py"
REFERENCE = REPO_ROOT / "docs" / "settings.md"


@pytest.fixture(scope="module")
def generator():
    """Import the generator by path; tools/ is not an installed package."""
    spec = importlib.util.spec_from_file_location("gen_settings_reference", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestReferenceIsCurrent:
    def test_committed_file_matches_the_generator(self, generator):
        assert REFERENCE.read_text(encoding="utf-8") == generator.render(), (
            "docs/settings.md is stale. Run `make docs-settings`."
        )

    def test_every_config_field_has_a_row(self):
        text = REFERENCE.read_text(encoding="utf-8")
        missing = [name for name in Config.model_fields if f"| `{name}` |" not in text]
        assert not missing, f"settings missing from the reference: {missing}"

    def test_every_setting_has_help_text(self, generator):
        # The generator refuses to run without it; assert the rule directly so a
        # failure names the setting rather than only failing `make lint`.
        missing = [name for name in Config.model_fields if not generator._help_text(name)]
        assert not missing, f"settings with no help text: {missing}"


class TestGeneratedFileIsMachineIndependent:
    """A committed generated file must not bake in one machine's answer.

    `chat_n_ctx_target` scales its default from host RAM. Rendering the number
    made docs/settings.md differ between a laptop and a CI runner, so the
    freshness check failed on a tree nobody had edited.
    """

    @staticmethod
    def _render_for_host(generator, total_gb: int) -> str:
        with patch("lilbee.core.system._read_total_memory_bytes", return_value=total_gb * 1024**3):
            return generator.render()

    def test_output_is_identical_on_a_small_and_a_large_host(self, generator):
        small = self._render_for_host(generator, 8)
        large = self._render_for_host(generator, 256)
        assert small == large, (
            "the reference varies with host memory. Add every setting whose default "
            "is computed from the host to HOST_SCALED in tools/gen_settings_reference.py."
        )

    def test_host_scaled_settings_render_the_rule_not_a_number(self, generator):
        assert generator.HOST_SCALED, "expected at least one host-scaled setting"
        for key, phrase in generator.HOST_SCALED.items():
            assert key in Config.model_fields
            assert generator._render_default(key) == phrase


class TestEnvironmentOnlyVariables:
    """Variables read from os.environ are invisible to Config-driven generation."""

    def test_every_env_var_read_in_src_is_documented_or_registered(self, generator):
        # Drive the generator's own scan rather than restating it, so the test
        # cannot pass while the check the build relies on is broken.
        generator._check_env_only_registry()

    def test_the_scan_rejects_an_unregistered_variable(self, generator, tmp_path, monkeypatch):
        """A check that cannot fail is not a check."""
        source = tmp_path / "src" / "lilbee" / "fake_module.py"
        source.parent.mkdir(parents=True)
        source.write_text('KNOB = "LILBEE_NOT_REGISTERED_ANYWHERE"\n', encoding="utf-8")
        monkeypatch.setattr(generator, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(generator, "ENV_ONLY", {})
        with pytest.raises(SystemExit, match="LILBEE_NOT_REGISTERED_ANYWHERE"):
            generator._check_env_only_registry()

    @pytest.mark.parametrize(
        "name",
        ["LILBEE_DATA", "LILBEE_LOG_LEVEL", "LILBEE_ENGINE_DIR", "LILBEE_TOKEN"],
    )
    def test_known_env_only_variables_reach_the_document(self, name):
        # These were documented in usage.md before the reference replaced its
        # tables. None is a Config field, so only the ENV_ONLY registry carries them.
        assert name in REFERENCE.read_text(encoding="utf-8")

    def test_internal_variables_stay_out_of_the_document(self, generator):
        text = REFERENCE.read_text(encoding="utf-8")
        internal = [n for n, (_, is_internal) in generator.ENV_ONLY.items() if is_internal]
        assert internal, "expected some test hooks to be registered but hidden"
        for name in internal:
            assert name not in text, f"{name} is a test hook and must not be documented"


class TestSurfaceColumns:
    """Each column is derived, so check the derivation against the boundary itself."""

    @pytest.mark.parametrize("key", sorted(MODEL_ROLE_FIELDS))
    def test_model_roles_are_http_role_api_not_patch_config(self, generator, key):
        # PATCH /api/config passes allow_model_roles=False, so the roles are only
        # settable through PUT /api/models/<role>.
        assert generator._http_cell(key) == "role API"
        assert generator._mcp_cell(key) == "yes"

    def test_write_only_keys_are_settable_but_not_listed(self, generator):
        write_only = sorted(set(WRITABLE_CONFIG_FIELDS) - PUBLIC_CONFIG_FIELDS)
        assert write_only, "expected the API keys to be write-only"
        for key in write_only:
            assert generator._mcp_cell(key) == "write-only"

    def test_fields_with_no_write_path_say_no(self, generator):
        unwritable = set(Config.model_fields) - set(WRITABLE_CONFIG_FIELDS) - MODEL_ROLE_FIELDS
        assert "server_port" in unwritable
        for key in unwritable:
            assert generator._mcp_cell(key) == "no"
            assert generator._http_cell(key) == "no"

    def test_cli_column_names_only_commands_that_exist(self, generator):
        # There is no general `lilbee set`; the column must not imply one.
        assert generator._cli_cell("top_k") == "no"
        assert "use-embedder" in generator._cli_cell("embedding_model")


class TestToolGateSettings:
    """The reference's reconnect rule is generated from TOOL_GATE_SETTINGS."""

    def test_gates_name_real_boolean_settings(self):
        for key in TOOL_GATE_SETTINGS:
            assert key in Config.model_fields, f"{key} is not a config field"
            assert Config.model_fields[key].annotation is bool

    @pytest.mark.parametrize("key", sorted(TOOL_GATE_SETTINGS))
    def test_each_gate_changes_the_registered_tool_list(self, monkeypatch, key):
        """A gate that stops gating makes the documented reconnect advice wrong."""
        monkeypatch.setattr(cfg, key, False)
        without = {tool.name for tool in build_mcp_server()._tool_manager.list_tools()}
        monkeypatch.setattr(cfg, key, True)
        with_it = {tool.name for tool in build_mcp_server()._tool_manager.list_tools()}
        assert with_it > without, f"{key} no longer gates any tool; update TOOL_GATE_SETTINGS"

    def test_wiki_status_registers_either_way(self, monkeypatch):
        """The reference promises a caller can always read the wiki's state."""
        for enabled in (False, True):
            monkeypatch.setattr(cfg, "wiki", enabled)
            names = {tool.name for tool in build_mcp_server()._tool_manager.list_tools()}
            assert "wiki_status" in names
            assert "wiki_wipe" in names
