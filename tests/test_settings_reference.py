"""The generated settings reference must keep agreeing with the real surfaces.

``docs/settings.md`` tells a caller whether MCP, HTTP, the TUI or the CLI can
change a given setting, and whether the change needs a reconnect or a reindex.
Those claims are only useful while they match the code, so they are asserted
here rather than trusted.
"""

from __future__ import annotations

import importlib.util
import sys
from enum import Enum
from pathlib import Path
from unittest.mock import patch

import pytest

from lilbee.config_meta import (
    ENV_UNSETTABLE_FIELDS,
    PUBLIC_CONFIG_FIELDS,
    SERVICE_ONLY_FIELDS,
    WRITABLE_CONFIG_FIELDS,
)
from lilbee.core.config import Config, cfg
from lilbee.core.config.model import _TAKES_DEFAULT_WHEN_REFUSED, _refusals
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

    def test_the_prose_names_each_setting_that_takes_its_default_when_refused(self):
        prose = REFERENCE.read_text(encoding="utf-8").split("## Reading the surface columns")[0]
        sentence = next(
            line for line in prose.splitlines() if "The exception is the settings" in line
        )
        clause = sentence.split("The exception is the settings")[1].split(":")[0]
        named = {name for name in Config.model_fields if f"`{name}`" in clause}
        assert len(named) == 4
        assert named == _TAKES_DEFAULT_WHEN_REFUSED


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


# Strings a variable could hold for some type of setting: numbers, flags, choices,
# lists, paths, model refs, URLs, and the shapes someone would try for a table.
_ENV_CANDIDATES = (
    "1", "0", "7", "0.5", "512", "true", "auto", "none", "all", "off", "a", "a,b", "eng",
    "/x", "{}", "[]", '{"a": "/x"}', "a=/x", "a:/x", "null", "org/repo/file.gguf",
    "http://localhost:1", "PERSON", "search", "english", "q8_0", "llm", "remote",
)  # fmt: skip


_SAMPLE_OVERRIDES = {
    # A valid value for each key whose built-in value is empty or None and needs a real shape.
    "placement": '{"chat": {"devices": [0]}}',
    "gpu_devices": "0",
    "num_ctx_max": "4096",
    "ocr_language": "deu",  # a Tesseract language code; any other text is refused
}

# Settings for which `lilbee settings set` alone accepts one text only, so no sample can differ
# from the value they hold. The reason is the refusal the CLI prints for any other text.
_VALUE_BEFORE_THE_SET: dict[str, object] = {
    # Every lilbee command starts by turning this off, so a sample of false would never differ.
    "worker_pool_eager_start": False,
}
_ONE_LEGAL_VALUE: dict[str, str] = {
    "crawl_password": "",  # a password needs a username, and one set call writes one setting
}


def _as_text(value: object) -> str:
    """The text a value would be typed as: a list is one item per line, a bool is lower case."""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if isinstance(value, Enum):
        return str(value.value)
    return str(value).lower() if isinstance(value, bool) else str(value)


def _candidate_samples(key: str, value: object) -> list[str]:
    """Texts that differ from *value*, most likely legal first, for the type of the field."""
    from lilbee.core.config.schema import field_type_name, field_value_set

    kind = field_type_name(key).split("|")[0]
    members = field_value_set(key)
    if members:
        return [member for member in members if member != _as_text(value)]
    if kind == "bool":
        return ["false" if value is True else "true"]
    if kind == "int":
        base = value if isinstance(value, int) else 0
        return [str(n) for n in (base + 1, base - 1, base * 2 + 1, 7)]
    if kind == "float":
        base = float(value) if isinstance(value, int | float) else 0.0
        return [str(n) for n in (round(base + 0.1, 3), round(base - 0.1, 3), base + 1, 0.5)]
    if kind == "list":
        return ["distinct-a\ndistinct-b", "7\n9", "distinct-a"]
    return ["distinct-value", "/distinct/path", "http://distinct.example", "7"]


def _sample_cli_value(key: str, current: object = None) -> str:
    """A text for `lilbee settings set <key>` that differs from the built-in and current values.

    A sample equal to the default lets a set path that drops the typed value pass.
    """
    from lilbee.core.config.resolve import builtin_value

    avoid = {_as_text(builtin_value(key)), _as_text(current)}
    if key in _ONE_LEGAL_VALUE:
        return _ONE_LEGAL_VALUE[key]
    if key in _SAMPLE_OVERRIDES:
        candidates = [_SAMPLE_OVERRIDES[key]]
    else:
        candidates = _candidate_samples(key, builtin_value(key) if current is None else current)
    fresh = [text for text in candidates if text not in avoid]
    accepted = [text for text in fresh if not _refusals(Config, {key: text})]
    return (accepted or fresh or candidates)[0]


class TestSurfaceColumns:
    """Each column is derived, so check the derivation against the boundary itself."""

    def test_the_settings_marked_as_set_by_no_variable_are_the_ones_no_string_sets(self):
        """Every setting against every candidate and its own default as text."""
        assert len(Config.model_fields) > 150
        unsettable = set()
        for key, info in Config.model_fields.items():
            default = str(info.get_default(call_default_factory=True))
            if all(_refusals(Config, {key: text}) for text in (*_ENV_CANDIDATES, default)):
                unsettable.add(key)
        assert unsettable == ENV_UNSETTABLE_FIELDS - SERVICE_ONLY_FIELDS == {"linked_roots"}

    def test_a_setting_no_variable_sets_shows_no_variable(self, generator):
        text = REFERENCE.read_text(encoding="utf-8")
        assert ENV_UNSETTABLE_FIELDS
        for key in ENV_UNSETTABLE_FIELDS:
            assert generator._env_cell(key) == "no"
            assert f"| `{key}` | no |" in text
            assert f"LILBEE_{key.upper()}" not in text
        assert generator._env_cell("top_k") == "`LILBEE_TOP_K`"
        assert "| `top_k` | `LILBEE_TOP_K` |" in text

    @pytest.mark.parametrize("key", sorted(MODEL_ROLE_FIELDS))
    def test_the_cli_column_of_a_model_role_reads_the_command_text_of_that_role(
        self, generator, key
    ):
        with patch.object(generator, "role_command", return_value="lilbee a-command <ref> now"):
            assert generator._cli_cell(key) == "`lilbee a-command REF now`"

    @pytest.mark.parametrize("key", sorted(MODEL_ROLE_FIELDS))
    def test_model_roles_are_http_role_api_not_patch_config(self, generator, key):
        # PATCH /api/config passes allow_model_roles=False, so the roles are only
        # settable through PUT /api/models/<role>.
        assert generator._http_cell(key) == "role API"
        assert generator._mcp_cell(key) == "yes"

    def test_write_only_keys_are_settable_but_not_listed(self, generator):
        write_only = sorted(
            set(WRITABLE_CONFIG_FIELDS) - PUBLIC_CONFIG_FIELDS - SERVICE_ONLY_FIELDS
        )
        assert write_only, "expected the API keys to be write-only"
        for key in write_only:
            assert generator._mcp_cell(key) == "write-only"

    @pytest.mark.parametrize("key", sorted(SERVICE_ONLY_FIELDS))
    def test_a_service_only_field_has_no_general_write_path(self, generator, key):
        assert key in WRITABLE_CONFIG_FIELDS
        assert generator._mcp_cell(key) == "no"
        assert generator._http_cell(key) == "no"

    def test_fields_with_no_write_path_say_no(self, generator):
        unwritable = set(Config.model_fields) - set(WRITABLE_CONFIG_FIELDS) - MODEL_ROLE_FIELDS
        assert "server_port" in unwritable
        for key in unwritable:
            assert generator._mcp_cell(key) == "no"
            assert generator._http_cell(key) == "no"

    def test_cli_column_lists_top_k_as_settable_from_the_cli(self, generator):
        row = next(
            line
            for line in REFERENCE.read_text(encoding="utf-8").splitlines()
            if line.startswith("| `top_k` |")
        )
        assert "`lilbee settings set`" in row
        assert generator._cli_cell("top_k") == "`lilbee settings set`"

    @pytest.mark.parametrize("key", sorted(Config.model_fields))
    def test_cli_column_matches_what_the_set_path_does(self, generator, tmp_path, monkeypatch, key):
        """Run `lilbee settings set` for every field and compare with the cell.

        An accepted set must leave the value in config.toml, and a refused one must
        leave the key out. One test for each key: a single loop over every key ran 15
        to 35 seconds alone and passed the 60 second test limit on a loaded host.
        """
        from typer.testing import CliRunner

        from lilbee.app import services as svc_mod
        from lilbee.cli.app import app
        from lilbee.core import settings as persistent_settings
        from tests.conftest import make_mock_services

        monkeypatch.delenv("LILBEE_DATA", raising=False)
        snapshot = cfg.model_copy()
        svc_mod.set_services(make_mock_services())
        before = _VALUE_BEFORE_THE_SET.get(key, getattr(snapshot, key))
        sample = _sample_cli_value(key, before)
        try:
            result = CliRunner().invoke(
                app,
                ["settings", "set", "--data-dir", str(tmp_path), "--", key, sample],
            )
            stored = persistent_settings.load(cfg.data_root)
            live = getattr(cfg, key)
        finally:
            svc_mod.set_services(None)
            for name in type(cfg).model_fields:
                setattr(cfg, name, getattr(snapshot, name))
        accepted = result.exit_code == 0
        marked = "`lilbee settings set`" in generator._cli_cell(key)
        assert accepted == marked, (
            f"the CLI column disagrees with `lilbee settings set` for {key}: "
            f"{'accepted' if accepted else 'refused'} {result.output[-90:]}"
        )
        if not accepted:
            assert key not in stored, f"a refused set of {key} wrote it to config.toml"
            return
        assert key in stored, f"`lilbee settings set {key}` exited 0 but config.toml lacks it"
        # config.toml keeps a list as one item per line and a path as text.
        on_disk = live if isinstance(live, bool | int | float | type(None)) else _as_text(live)
        assert stored[key] == on_disk, f"config.toml holds {stored[key]!r}, expected {on_disk!r}"
        assert _as_text(live) == sample, f"`lilbee settings set {key} {sample!r}` stored {live!r}"
        assert key in _ONE_LEGAL_VALUE or sample != _as_text(before), (
            f"the sample for {key} equals its value before the set, so a dropped value would pass"
        )

    def test_cli_column_keeps_the_dedicated_commands(self, generator):
        assert "use-embedder" in generator._cli_cell("embedding_model")
        assert "placement set" in generator._cli_cell("placement")
        assert generator._cli_cell("chat_model") == "`lilbee model use chat REF`"
        for key in SERVICE_ONLY_FIELDS:
            assert generator._cli_cell(key) == "no"
        assert generator._cli_cell("linked_roots") == "`lilbee add`, `lilbee remove`"

    def test_cli_column_says_no_for_a_key_the_cli_cannot_write(self, generator):
        assert generator._cli_cell("server_port") == "no"

    def test_the_prose_names_the_settings_commands_and_no_missing_set_command(self, generator):
        text = generator.render()
        assert "no general `lilbee set` command" not in text
        assert "| CLI | `lilbee settings set <setting> <value>`" in text
        for command in ("lilbee settings list", "lilbee settings unset"):
            assert command in text

    def test_a_setting_on_a_dedicated_tab_names_that_tab_in_the_tui_column(self, generator):
        from lilbee.app.settings_map import SETTINGS_MAP
        from lilbee.cli.tui import messages as msg

        assert SETTINGS_MAP["choose_engine_on_pick"].tab == msg.SETTINGS_ENGINES_TAB
        assert generator._tui_cell("choose_engine_on_pick") == "Engines tab"
        assert "| `choose_engine_on_pick` |" in generator.render()
        for key, defn in SETTINGS_MAP.items():
            if defn.tab is not None:
                assert generator._tui_cell(key) == f"{defn.tab} tab"
            elif defn.hidden:
                assert generator._tui_cell(key) == "no"

    def test_advanced_column_matches_the_settings_map_flag(self, generator):
        from lilbee.app.settings_map import SETTINGS_MAP

        for key, defn in SETTINGS_MAP.items():
            expected = "yes" if defn.advanced else "no"
            assert generator._advanced_cell(key) == expected
        # A field with no SettingDef entry (env/config-file only) is never advanced.
        for key in set(Config.model_fields) - set(SETTINGS_MAP):
            assert generator._advanced_cell(key) == "no"


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
