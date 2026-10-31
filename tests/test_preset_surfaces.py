"""Parity of the preset operations across the service, CLI, REST, MCP and the slash command.

The service is the oracle. Each case runs on the service first, on its own fresh store. Every
other surface then runs the same input on a fresh store, and its result is compared with what the
service itself returned, never with a table of expected texts. The stored list is also compared
with a list typed in the case, so the service cannot drift together with its surfaces.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import lilbee.mcp_server as mcp_server
from lilbee.app.presets import (
    add_preset,
    edit_preset,
    failure_text,
    list_presets,
    remove_preset,
)
from lilbee.cli import app as cli_app
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.preset_commands import run_preset_command
from lilbee.core import settings
from lilbee.core.config import Config, cfg
from lilbee.server.models import PresetListResponse, PresetRemovedResponse, PresetResponse
from lilbee.server.routes import presets as routes
from tests._litestar_test_client import create_test_client

KEY = "SENTINEL-parity-key-3f9a"
NEW_KEY = "SENTINEL-parity-new-key-81c2"
LOCAL = "http://127.0.0.1:8000/v1"
SAME_ORIGIN = "http://127.0.0.1:8000/v2"
OTHER = "http://127.0.0.1:9000/v1"
OPERATIONS = ("list", "add", "edit", "remove")
DATA_SURFACES = ("cli", "rest", "mcp")
SURFACES = (*DATA_SURFACES, "slash")
ALL_ROUTES = [
    routes.presets_list_route,
    routes.presets_add_route,
    routes.presets_edit_route,
    routes.presets_remove_route,
]

Args = dict[str, Any]


@dataclass(frozen=True)
class Outcome:
    """What one surface did: whether it worked, its data (or the line it showed), its error."""

    ok: bool
    data: Any
    error: str


@dataclass(frozen=True)
class Case:
    """One operation on a seeded store, with the store it must leave behind."""

    id: str
    op: str
    seed: tuple[Args, ...]
    args: Args
    ok: bool
    stored: list[dict[str, Any]]
    key_given: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_given", "key" in self.args)


OMLX_SEED: Args = {"name": "omlx", "url": LOCAL, "roles": ["chat", "embed"], "key": KEY}
OMLX_STORED = {"name": "omlx", "url": LOCAL, "roles": ["chat", "embed"], "api_key": KEY}
OMLX_NO_KEY = {"name": "omlx", "url": LOCAL, "roles": ["chat", "embed"]}
ADD: Args = {"name": "omlx", "url": LOCAL, "roles": ["chat", "embed"]}


def _refused_add(case_id: str, **change: Any) -> Case:
    return Case(case_id, "add", (), {**ADD, **change}, ok=False, stored=[])


CASES: tuple[Case, ...] = (
    Case("list-empty", "list", (), {}, ok=True, stored=[]),
    Case("list-one-with-key", "list", (OMLX_SEED,), {}, ok=True, stored=[OMLX_STORED]),
    Case(
        "add-valid",
        "add",
        (),
        {"name": "rr", "url": LOCAL, "roles": ["rerank"], "wire": "tei"},
        ok=True,
        stored=[{"name": "rr", "url": LOCAL, "roles": ["rerank"], "rerank_wire": "tei"}],
    ),
    Case("add-valid-with-key", "add", (), {**ADD, "key": KEY}, ok=True, stored=[OMLX_STORED]),
    Case("add-name-taken", "add", (OMLX_SEED,), ADD, ok=False, stored=[OMLX_STORED]),
    _refused_add("add-bad-name", name="Bad Name"),
    _refused_add("add-reserved-name", name="openai"),
    _refused_add("add-no-role", roles=[]),
    _refused_add("add-unknown-role", roles=["frob"]),
    _refused_add("add-repeated-role", roles=["chat", "chat"]),
    _refused_add("add-bad-scheme", url="ftp://127.0.0.1/"),
    _refused_add("add-credentials", url="http://user:pass@127.0.0.1:8000/v1"),
    _refused_add("add-blocked-address", url="http://169.254.169.254/"),
    _refused_add("add-unknown-wire", roles=["rerank"], wire="v9"),
    Case(
        "edit-url-same-origin",
        "edit",
        (OMLX_SEED,),
        {"name": "omlx", "url": SAME_ORIGIN},
        ok=True,
        stored=[{**OMLX_STORED, "url": SAME_ORIGIN}],
    ),
    Case(
        "edit-url-new-origin-no-new-key",
        "edit",
        (OMLX_SEED,),
        {"name": "omlx", "url": OTHER},
        ok=False,
        stored=[OMLX_STORED],
    ),
    Case(
        "edit-roles-and-wire",
        "edit",
        (OMLX_SEED,),
        {"name": "omlx", "roles": ["embed", "rerank"], "wire": "tei"},
        ok=True,
        stored=[{**OMLX_STORED, "roles": ["embed", "rerank"], "rerank_wire": "tei"}],
    ),
    Case(
        "edit-new-key",
        "edit",
        (OMLX_SEED,),
        {"name": "omlx", "key": NEW_KEY},
        ok=True,
        stored=[{**OMLX_STORED, "api_key": NEW_KEY}],
    ),
    Case(
        "edit-clear-key",
        "edit",
        (OMLX_SEED,),
        {"name": "omlx", "clear_key": True},
        ok=True,
        stored=[OMLX_NO_KEY],
    ),
    Case(
        "edit-clear-key-with-another-field",
        "edit",
        (OMLX_SEED,),
        {"name": "omlx", "clear_key": True, "wire": "tei"},
        ok=False,
        stored=[OMLX_STORED],
    ),
    Case("edit-no-field", "edit", (OMLX_SEED,), {"name": "omlx"}, ok=False, stored=[OMLX_STORED]),
    Case(
        "edit-missing",
        "edit",
        (OMLX_SEED,),
        {"name": "nope", "wire": "tei"},
        ok=False,
        stored=[OMLX_STORED],
    ),
    Case("remove-valid", "remove", (OMLX_SEED,), {"name": "omlx"}, ok=True, stored=[]),
    Case(
        "remove-missing", "remove", (OMLX_SEED,), {"name": "nope"}, ok=False, stored=[OMLX_STORED]
    ),
)


def _stored() -> list[dict[str, Any]]:
    raw = settings.load(cfg.data_root).get("server_presets")
    return json.loads(raw) if raw else []


@pytest.fixture(autouse=True)
def _restore_config(monkeypatch):
    snapshot = cfg.model_copy()
    cfg.allow_http_engines = True
    cfg.mcp_presets_enabled = True
    yield
    mcp_server.set_http_mounted(False)
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _fresh_store(root: Path, seed: tuple[Args, ...], monkeypatch) -> None:
    """Point the config at a new empty directory, then store the seed through the service."""
    root.mkdir(parents=True)
    monkeypatch.setenv("LILBEE_DATA", str(root))
    cfg.data_root = root
    cfg.server_presets = None
    for entry in seed:
        add_preset(entry["name"], entry["url"], roles=entry["roles"], api_key=entry.get("key"))


@pytest.fixture
def client() -> Iterator[Any]:
    with create_test_client(ALL_ROUTES) as cl:
        yield cl


def _as_data(model: Any) -> Any:
    return model.model_dump(mode="json")


def service(op: str, args: Args) -> Outcome:
    """The oracle: the service called directly, shaped as the three data surfaces return it."""
    try:
        if op == "list":
            views = [PresetResponse.from_view(p) for p in list_presets()]
            data = _as_data(PresetListResponse(presets=views))
        elif op == "add":
            data = _as_data(
                PresetResponse.from_view(
                    add_preset(
                        args["name"],
                        args["url"],
                        roles=args["roles"],
                        api_key=args.get("key"),
                        rerank_wire=args.get("wire", "v1"),
                    )
                )
            )
        elif op == "edit":
            data = _as_data(
                PresetResponse.from_view(
                    edit_preset(
                        args["name"],
                        url=args.get("url"),
                        roles=args.get("roles"),
                        api_key=args.get("key"),
                        rerank_wire=args.get("wire"),
                        clear_key=args.get("clear_key", False),
                    )
                )
            )
        else:
            remove_preset(args["name"])
            data = _as_data(PresetRemovedResponse(removed=args["name"]))
    except (ValueError, OSError) as exc:
        return Outcome(False, None, failure_text(exc))
    return Outcome(True, data, "")


def _cli_argv(op: str, args: Args) -> list[str]:
    """The words after `lilbee --json preset` for *op*; the key flag is added by the caller."""
    if op == "list":
        return ["list"]
    if op == "remove":
        return ["remove", args["name"], "--yes"]
    if op == "add":
        return [
            "add",
            args["name"],
            args["url"],
            "--roles",
            ",".join(args["roles"]),
            "--wire",
            args.get("wire", "v1"),
        ]
    argv = ["edit", args["name"]]
    for flag, field_name in (("--url", "url"), ("--wire", "wire")):
        if field_name in args:
            argv += [flag, args[field_name]]
    if "roles" in args:
        argv += ["--roles", ",".join(args["roles"])]
    if args.get("clear_key"):
        argv.append("--clear-key")
    return argv


def cli(op: str, args: Args) -> Outcome:
    """`lilbee --json preset ...` through the root app; the key goes in on stdin."""
    argv = ["--json", "preset", *_cli_argv(op, args)]
    stdin = None
    if "key" in args:
        argv.append("--key-stdin")
        stdin = args["key"] + "\n"
    result = CliRunner().invoke(cli_app, argv, input=stdin)
    body = json.loads(result.stdout.strip().splitlines()[-1])
    if result.exit_code == 0:
        return Outcome(True, body, "")
    return Outcome(False, None, body["error"])


def rest(client: Any, op: str, args: Args) -> Outcome:
    """The REST routes through a test client; the key goes in the body."""
    if op == "list":
        res = client.get("/api/presets")
    elif op == "add":
        body = {k: args[k] for k in ("name", "url", "roles") if k in args}
        body.update({"api_key": args["key"]} if "key" in args else {})
        body.update({"rerank_wire": args["wire"]} if "wire" in args else {})
        res = client.post("/api/presets", json=body)
    elif op == "edit":
        body = {k: args[k] for k in ("url", "roles", "clear_key") if k in args}
        body.update({"api_key": args["key"]} if "key" in args else {})
        body.update({"rerank_wire": args["wire"]} if "wire" in args else {})
        res = client.put(f"/api/presets/{args['name']}", json=body)
    else:
        res = client.delete(f"/api/presets/{args['name']}")
    if res.status_code < 400:
        return Outcome(True, res.json(), "")
    return Outcome(False, None, res.json()["detail"])


def mcp(op: str, args: Args) -> Outcome:
    """The MCP tool functions, called as the server calls them."""
    if op == "list":
        body = mcp_server.preset_list_tool()
    elif op == "add":
        body = mcp_server.preset_add_tool(
            args["name"],
            args["url"],
            args["roles"],
            api_key=args.get("key"),
            rerank_wire=args.get("wire", "v1"),
        )
    elif op == "edit":
        body = mcp_server.preset_edit_tool(
            args["name"],
            url=args.get("url"),
            roles=args.get("roles"),
            api_key=args.get("key"),
            rerank_wire=args.get("wire"),
            clear_key=args.get("clear_key", False),
        )
    else:
        body = mcp_server.preset_remove_tool(args["name"])
    if "error" in body:
        return Outcome(False, None, body["error"])
    return Outcome(True, body, "")


def slash(op: str, args: Args) -> Outcome:
    """The /preset text command. It has no key entry, so a key is typed as the flag it refuses."""
    q = shlex.quote
    if op == "list":
        line = "list"
    elif op == "add":
        line = f"add {q(args['name'])} {q(args['url'])} --roles {q(','.join(args['roles']))}"
        line += f" --wire {q(args.get('wire', 'v1'))}"
    elif op == "edit":
        line = f"edit {q(args['name'])}"
        for flag, field_name in (("--url", "url"), ("--wire", "wire")):
            if field_name in args:
                line += f" {flag} {q(args[field_name])}"
        if "roles" in args:
            line += f" --roles {q(','.join(args['roles']))}"
        if args.get("clear_key"):
            line += " --clear-key"
    else:
        line = f"remove {q(args['name'])}"
    if "key" in args:
        line += f" --key {q(args['key'])}"
    result = run_preset_command(line)
    return Outcome(result.ok, result.text if result.ok else None, "" if result.ok else result.text)


def _slash_line_for(oracle: Outcome, op: str) -> str:
    """The line the slash command shows for what the oracle returned."""
    if op == "list":
        presets = oracle.data["presets"]
        if not presets:
            return msg.PRESET_LIST_EMPTY
        return "\n".join(
            msg.SETTINGS_PRESETS_ROW.format(
                name=p["name"],
                url=p["base_url"] or msg.PRESET_INVALID_URL,
                roles=", ".join(p["roles"]) or msg.PRESET_NO_VALUE,
                wire=p["rerank_wire"] if "rerank" in p["roles"] else msg.PRESET_NO_VALUE,
                key=msg.PRESET_KEY_SET if p["has_key"] else msg.PRESET_KEY_NONE,
            )
            for p in presets
        )
    if op == "add":
        return msg.PRESET_ADDED.format(name=oracle.data["name"])
    if op == "edit":
        return msg.PRESET_UPDATED.format(name=oracle.data["name"])
    return msg.PRESET_REMOVED.format(name=oracle.data["removed"])


def _run(surface: str, client: Any, case: Case) -> Outcome:
    if surface == "cli":
        return cli(case.op, case.args)
    if surface == "rest":
        return rest(client, case.op, case.args)
    if surface == "mcp":
        return mcp(case.op, case.args)
    return slash(case.op, case.args)


def _ids() -> list[str]:
    return [case.id for case in CASES]


class TestTheOracleMatchesTheCases:
    @pytest.mark.parametrize("case", CASES, ids=_ids())
    def test_the_service_leaves_the_stored_list_the_case_types(self, case, tmp_path, monkeypatch):
        _fresh_store(tmp_path / "service", case.seed, monkeypatch)
        outcome = service(case.op, case.args)
        assert outcome.ok is case.ok
        assert bool(outcome.error) is (not case.ok)
        assert _stored() == case.stored


class TestEverySurfaceMatchesTheService:
    @pytest.mark.parametrize("surface", SURFACES)
    @pytest.mark.parametrize("case", CASES, ids=_ids())
    def test_same_result_error_text_and_stored_list(
        self, case, surface, tmp_path, monkeypatch, client
    ):
        _fresh_store(tmp_path / "service", case.seed, monkeypatch)
        before = _stored()
        oracle = service(case.op, case.args)
        oracle_stored = _stored()

        _fresh_store(tmp_path / surface, case.seed, monkeypatch)
        outcome = _run(surface, client, case)

        if surface == "slash" and case.key_given:
            assert outcome == Outcome(False, None, msg.PRESET_KEY_IN_DIALOG)
            assert _stored() == before
        else:
            if surface == "slash" and oracle.ok:
                assert outcome == Outcome(True, _slash_line_for(oracle, case.op), "")
            else:
                assert outcome.ok is oracle.ok
                assert outcome.error == oracle.error
            if surface in DATA_SURFACES:
                assert outcome.data == oracle.data
            assert _stored() == oracle_stored == case.stored
        for secret in (KEY, NEW_KEY):
            assert secret not in repr(outcome)


class TestTheCaseTable:
    def test_the_table_is_nonempty_and_has_unique_ids(self):
        assert len(CASES) > 0
        assert len(set(_ids())) == len(CASES)

    def test_every_operation_has_a_working_and_a_refused_case(self):
        for op in OPERATIONS:
            oks = {case.ok for case in CASES if case.op == op}
            assert oks == ({True} if op == "list" else {True, False}), op

    def test_key_cases_exist_for_add_and_edit(self):
        keyed = {case.op for case in CASES if case.key_given}
        assert keyed == {"add", "edit"}
