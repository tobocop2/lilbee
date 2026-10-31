"""Tests for the `lilbee preset` CLI sub-app, run against the real service and a temp config."""

import json
import types

import pytest
from typer.testing import CliRunner

import lilbee.cli.preset as cli_preset
from lilbee.cli import app
from lilbee.core import settings
from lilbee.core.config import Config, cfg

runner = CliRunner()
KEY = "SENTINEL-cli-key-77a3"
LOCAL = "http://127.0.0.1:8000/v1"
OTHER = "http://127.0.0.1:9000/v1"


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    snapshot = cfg.model_copy()
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    cfg.data_root = tmp_path
    cfg.server_presets = None
    yield
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _stored() -> list[dict]:
    raw = settings.load(cfg.data_root).get("server_presets")
    return json.loads(raw) if raw else []


def _json(*argv: str, input: str | None = None):
    """Run `lilbee --json preset ...`; return the result and its last JSON line."""
    result = runner.invoke(app, ["--json", "preset", *argv], input=input)
    return result, json.loads(result.stdout.strip().splitlines()[-1])


def _seed() -> None:
    result, _ = _json(
        "add", "omlx", LOCAL, "--roles", "chat,embed", "--key-stdin", input=KEY + "\n"
    )
    assert result.exit_code == 0


class TestJson:
    def test_list_with_none_is_an_empty_list(self):
        result, body = _json("list")
        assert (result.exit_code, body) == (0, {"presets": []})

    def test_add_prints_the_view_and_stores_the_preset(self):
        result, body = _json("add", "omlx", LOCAL, "--roles", "chat, embed")
        assert result.exit_code == 0
        assert body == {
            "name": "omlx",
            "base_url": LOCAL,
            "roles": ["chat", "embed"],
            "has_key": False,
            "valid": True,
            "rerank_wire": "v1",
        }
        assert _stored() == [{"name": "omlx", "url": LOCAL, "roles": ["chat", "embed"]}]

    def test_add_stores_the_wire(self):
        _, body = _json("add", "rr", LOCAL, "--roles", "rerank", "--wire", "tei")
        assert body["rerank_wire"] == "tei"
        assert _stored()[0]["rerank_wire"] == "tei"

    def test_the_key_comes_from_stdin_and_never_prints(self):
        result, body = _json(
            "add", "omlx", LOCAL, "--roles", "chat", "--key-stdin", input=KEY + "\n"
        )
        assert body["has_key"] is True
        assert KEY not in result.output
        assert _stored()[0]["api_key"] == KEY

    def test_list_shows_the_key_state_and_never_the_key(self):
        _seed()
        result, body = _json("list")
        assert [p["has_key"] for p in body["presets"]] == [True]
        assert KEY not in result.output

    def test_edit_changes_what_is_given_and_keeps_the_rest(self):
        _seed()
        _, body = _json("edit", "omlx", "--roles", "chat", "--wire", "tei")
        assert (body["roles"], body["rerank_wire"], body["has_key"]) == (["chat"], "tei", True)
        assert _stored()[0]["url"] == LOCAL

    def test_edit_with_a_new_key_from_stdin(self):
        _seed()
        _, body = _json("edit", "omlx", "--url", OTHER, "--key-stdin", input="NEW-" + KEY + "\n")
        assert body["base_url"] == OTHER
        assert _stored()[0]["api_key"] == "NEW-" + KEY

    def test_clear_key_removes_the_key(self):
        _seed()
        result, body = _json("edit", "omlx", "--clear-key")
        assert (result.exit_code, body["has_key"]) == (0, False)
        assert "api_key" not in _stored()[0]

    def test_remove_with_yes(self):
        _seed()
        result, body = _json("remove", "omlx", "--yes")
        assert (result.exit_code, body) == (0, {"removed": "omlx"})
        assert _stored() == []


class TestRefusals:
    """The CLI prints the service's text and exits 1."""

    @pytest.mark.parametrize(
        ("argv", "text"),
        [
            (["add", "omlx", LOCAL, "--roles", "chat"], "A server preset named 'omlx' already"),
            (["add", "Bad Name", LOCAL, "--roles", "chat"], "A preset name uses lower-case"),
            (["add", "x1", LOCAL, "--roles", ""], "A preset needs at least one role"),
            (["add", "x1", LOCAL, "--roles", "chatt"], "Unknown role 'chatt'"),
            (["add", "x1", LOCAL, "--roles", "chat,,embed"], "Unknown role ''"),
            (["add", "x1", LOCAL, "--roles", "chat", "--wire", "cohere"], "Unknown rerank wire"),
            (["edit", "nope", "--url", LOCAL], "No server preset is named 'nope'"),
            (["edit", "omlx"], "Nothing to change"),
            (["edit", "omlx", "--clear-key", "--wire", "tei"], "Clear the key in its own edit"),
            (["edit", "omlx", "--url", "http://127.0.0.2:1/v1"], "stored key does not follow"),
            (["remove", "nope", "--yes"], "No server preset is named 'nope'"),
        ],
    )
    def test_json_error_and_exit_1(self, argv, text):
        _seed()
        result, body = _json(*argv)
        assert result.exit_code == 1
        assert set(body) == {"error"}
        assert text in body["error"]
        assert KEY not in result.output

    def test_a_refusal_leaves_the_store_as_it_was(self):
        _seed()
        before = _stored()
        _json("edit", "omlx", "--roles", "chatt")
        assert _stored() == before

    def test_the_human_form_prints_the_text_and_exits_1(self):
        result = runner.invoke(app, ["preset", "edit", "nope", "--url", LOCAL])
        assert result.exit_code == 1
        assert "No server preset is named 'nope'" in result.output

    def test_a_write_failure_prints_the_config_write_text(self, monkeypatch):
        def locked(*args, **kwargs):
            raise PermissionError(13, "denied")

        monkeypatch.setattr(cli_preset, "list_presets", locked)
        result, body = _json("list")
        assert result.exit_code == 1
        assert "Could not write config.toml" in body["error"]


class TestHuman:
    def test_an_empty_list_says_so(self):
        result = runner.invoke(app, ["preset", "list"])
        assert result.exit_code == 0
        assert "No server presets." in result.output

    def test_the_list_table_shows_name_url_roles_wire_and_key_state(self):
        _seed()
        runner.invoke(app, ["preset", "add", "rr", OTHER, "--roles", "rerank", "--wire", "tei"])
        result = runner.invoke(app, ["preset", "list"], terminal_width=120)
        assert result.exit_code == 0
        rows = {
            line.split("│")[1].strip(): line for line in result.output.splitlines() if "│" in line
        }
        assert LOCAL in rows["omlx"]
        assert "chat, embed" in rows["omlx"]
        assert "set" in rows["omlx"]
        assert "tei" in rows["rr"]
        assert "none" in rows["rr"]
        assert KEY not in result.output

    def test_the_wire_shows_a_dash_without_the_rerank_role(self):
        runner.invoke(app, ["preset", "add", "c", LOCAL, "--roles", "chat", "--wire", "tei"])
        result = runner.invoke(app, ["preset", "list"], terminal_width=120)
        row = next(line for line in result.output.splitlines() if "│ c " in line)
        assert "tei" not in row

    def test_an_invalid_stored_entry_is_marked(self):
        settings.set_value(
            cfg.data_root,
            "server_presets",
            json.dumps([{"name": "bad", "url": "ftp://x.test/", "roles": ["chat"]}]),
        )
        result = runner.invoke(app, ["preset", "list"], terminal_width=120)
        assert "(invalid)" in result.output

    def test_add_edit_and_remove_say_what_they_did(self):
        added = runner.invoke(app, ["preset", "add", "omlx", LOCAL, "--roles", "chat"])
        edited = runner.invoke(app, ["preset", "edit", "omlx", "--roles", "embed"])
        removed = runner.invoke(app, ["preset", "remove", "omlx", "--yes"])
        assert "Added server preset 'omlx'." in added.output
        assert "Updated server preset 'omlx'." in edited.output
        assert "Removed server preset 'omlx'." in removed.output


class TestRemoveAsks:
    def test_no_keeps_the_preset(self):
        _seed()
        result = runner.invoke(app, ["preset", "remove", "omlx"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted." in result.output
        assert [e["name"] for e in _stored()] == ["omlx"]

    def test_yes_at_the_prompt_removes_it(self):
        _seed()
        result = runner.invoke(app, ["preset", "remove", "omlx"], input="y\n")
        assert result.exit_code == 0
        assert _stored() == []

    def test_the_yes_flag_skips_the_prompt(self):
        _seed()
        result = runner.invoke(app, ["preset", "remove", "omlx", "-y"])
        assert (result.exit_code, _stored()) == (0, [])


class TestKeyEntry:
    def test_no_key_flag_exists(self):
        result = runner.invoke(app, ["preset", "add", "x1", LOCAL, "--roles", "chat", "--key", KEY])
        assert result.exit_code == 2
        assert "No such option" in result.output
        assert _stored() == []

    def test_a_terminal_gets_a_hidden_prompt(self, monkeypatch):
        tty = types.SimpleNamespace(stdin=types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr(cli_preset, "sys", tty)
        result = runner.invoke(
            app,
            ["preset", "add", "omlx", LOCAL, "--roles", "chat", "--key-stdin"],
            input=KEY + "\n",
        )
        assert result.exit_code == 0
        assert _stored()[0]["api_key"] == KEY
        assert KEY not in result.output

    def test_an_empty_line_on_stdin_is_no_key(self):
        _json("add", "omlx", LOCAL, "--roles", "chat", "--key-stdin", input="\n")
        assert "api_key" not in _stored()[0]

    def test_a_missing_url_is_a_usage_error(self):
        result = runner.invoke(app, ["preset", "add", "x1"])
        assert result.exit_code == 2
        assert "Missing argument" in result.output
