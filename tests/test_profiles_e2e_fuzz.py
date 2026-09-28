"""Untrusted profile input through the CLI, HTTP and MCP: generated names, files and folders.

Every input either succeeds or gets a user-facing refusal, never a traceback, and no
surface writes anywhere but the profile folders.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import quote

import pytest
import tomli_w
from hypothesis import HealthCheck, Phase, event, example, given, settings
from hypothesis import strategies as st
from litestar.testing import TestClient
from typer.testing import Result

from lilbee.app import analyze as analyze_mod
from lilbee.core.config import cfg
from lilbee.core.config.resolve import PROFILE_FIELDS
from lilbee.core.profile_files import (
    MAX_PROFILE_BYTES,
    PROFILE_SUFFIX,
    PROFILES_DIRNAME,
    RESERVED_NAME_KEYS,
    profile_key,
)
from lilbee.server.routes.profiles import PROFILE_BODY_MAX_BYTES
from tests._profiles_e2e import (
    GERMAN,
    TRACEBACK,
    World,
    cli,
    enter,
    http_client,
    mcp_call,
    real_root_listing,
)

F = TypeVar("F", bound=Callable[..., Any])

# No shrink phase: a failing example is reported as found, well inside the 60 s timeout.
FUZZ = settings(
    max_examples=20,
    deadline=dt.timedelta(seconds=10),
    derandomize=True,
    database=None,
    phases=[Phase.explicit, Phase.generate],
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
# HTTP refusals: 400 bad input, 404 no such profile, 409 a name clash, 413 too large.
REFUSED_STATUSES = frozenset({400, 404, 409, 413})
# The fullwidth forms of "Court", which casefold to their own letters, not to ASCII.
FULLWIDTH_COURT = "".join(chr(ord(letter) + 0xFEE0) for letter in "Court")
HOSTILE_NAMES = (
    "",
    " ",
    "..",
    "../evil",
    "a/b",
    "a\\b",
    "con",
    "con.toml",
    "LPT1",
    "COM9",
    "nul",
    "new",
    "active",
    "discard",
    "import",
    "validate",
    "Default",
    "Scanned archive",
    "x" * 41,
    FULLWIDTH_COURT,
    "()",
    "--help",
    "-x",
    "tab\tname",
    "line\nbreak",
    "nul\x00byte",
)
VALID_TEXT = '[profile]\nname = "Plain upload"\n[values]\ntop_k = 7\n'
HOSTILE_TEXTS = (
    "",
    "not toml at all [",
    "[values]\na = " + "[" * 400 + "]" * 400 + "\n",
    "[values]\n# " + "x" * (MAX_PROFILE_BYTES + 1) + "\n",
    "\ufeff" + VALID_TEXT,
    '[profile]\nname = "con"\n[values]\n',
    '[profile]\nname = "../../evil"\n[values]\n',
    "[values]\nchat_model = 'x'\ndata_root = '/tmp'\n",
    "[values]\ntop_k = 'nine'\nchunk_size = -1\n",
)
HOSTILE_BYTES = (
    b"\xff\xfe[values]\n",
    b"\x00\x01\x02",
    ("\ufeff" + VALID_TEXT).encode("utf-8"),
    b"[values]\n# " + b"x" * (MAX_PROFILE_BYTES + 1),
)
HOSTILE_FILENAMES = (
    "../../evil.toml",
    "..\\..\\evil.toml",
    "a/b.toml",
    "C:\\x.toml",
    "/etc/passwd.toml",
    ".toml",
    "",
    "con.toml",
    "LPT1.toml",
    "x" * 300 + ".toml",
)
# Folder specs: <world> stands for the world's folder, which only exists inside a test.
WORLD = "<world>"
HOSTILE_FOLDERS = (
    f"{WORLD}/notes",
    f"{WORLD}/inbox",
    f"{WORLD}/inbox/plain.txt",
    f"{WORLD}/link",
    f"{WORLD}/absent/x",
    "",
    " ",
    "..",
    "relative/dir",
    "~/nope-lilbee-fuzz",
    "C:\\nope",
    "nul\x00byte",
)
# Names other xdist workers and sessions create next to this test's folders.
_PYTEST_OWN = re.compile(r"(popen-gw\d+|pytest-\d+|pytest-current|garbage-.*|.*\.lock)")
# The folders pytest creates above a test's own; above them other processes write freely.
_PYTEST_LEVEL = re.compile(r"(popen-gw\d+|pytest-\d+|pytest-of-.+)")

_names = st.text(max_size=60)
_scalars = st.one_of(
    st.booleans(),
    st.integers(min_value=-(2**63), max_value=2**63 - 1),
    st.floats(),
    st.text(max_size=30),
    st.datetimes(min_value=dt.datetime(1, 1, 1), max_value=dt.datetime(9999, 12, 31)),
)
_toml_values = st.recursive(
    _scalars,
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=8), inner),
    max_leaves=12,
)
_value_keys = st.one_of(
    st.sampled_from(sorted(PROFILE_FIELDS)),
    st.sampled_from(["chat_model", "top_kk", "data_root", "api_key"]),
    st.text(max_size=12),
)
_meta = st.dictionaries(
    st.sampled_from(["name", "description", "authors", "format", "min_lilbee", "evidence", "x"]),
    _toml_values,
    max_size=4,
)
_documents = st.fixed_dictionaries(
    {},
    optional={
        "profile": _meta,
        "values": st.dictionaries(_value_keys, _toml_values, max_size=6),
        "extra": _toml_values,
    },
)
_valid_documents = st.fixed_dictionaries(
    {
        "profile": st.fixed_dictionaries(
            {"name": st.from_regex(r"[A-Za-z][A-Za-z0-9 _-]{0,20}", fullmatch=True)}
        ),
        "values": st.fixed_dictionaries(
            {}, optional={"top_k": st.integers(1, 50), "hyde": st.booleans()}
        ),
    }
)
_texts = st.one_of(
    _documents.map(tomli_w.dumps), _valid_documents.map(tomli_w.dumps), st.text(max_size=200)
)
_raw_bytes = st.one_of(_texts.map(lambda text: text.encode("utf-8")), st.binary(max_size=200))
_filenames = st.text(max_size=30)
_folder_words = st.text(
    alphabet=st.characters(codec="utf-8", exclude_characters="./\\\x00"), max_size=20
)
_folders = st.one_of(_folder_words, _folder_words.map(lambda word: f"{WORLD}/absent/{word}"))


def _examples(**cases: tuple[Any, ...]) -> Callable[[F], F]:
    """Run every listed value of each argument as an explicit example, on every run."""

    def _decorate(fn: F) -> F:
        for argname, values in cases.items():
            for value in values:
                fn = example(**{argname: value})(fn)
        return fn

    return _decorate


def _upload_examples(fn: F) -> F:
    """Each hostile file name with a valid file, and each hostile file with a plain name."""
    for filename in HOSTILE_FILENAMES:
        fn = example(text=VALID_TEXT, filename=filename)(fn)
    for text in HOSTILE_TEXTS:
        fn = example(text=text, filename="upload.toml")(fn)
    return fn


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _toml_listing(root: Path, skip: Path) -> dict[str, tuple[int, int]]:
    """Size and mtime of every TOML file under *root* outside *skip*.

    Profile and config writes are TOML; other tests' late log or index writes are not.
    """
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob(f"*{PROFILE_SUFFIX}"))
        if path.is_file() and not path.is_relative_to(skip)
    }


def _foreign_names(folder: Path) -> set[str]:
    """Names in *folder* that pytest and xdist did not create."""
    return {path.name for path in folder.iterdir() if not _PYTEST_OWN.fullmatch(path.name)}


class _Guard:
    """Fails when a surface writes outside the profile folders of its world.

    Each example checks the test's own folder byte for byte, the folder beside it and the
    real global root; the rest of this worker's folders are checked once, at teardown.
    """

    def __init__(self, world: World, bound: Path) -> None:
        self._bound = bound
        self._allowed = (
            world.global_profiles().resolve(),
            (world.root / PROFILES_DIRNAME).resolve(),
        )
        self._real = real_root_listing()
        self._before = _tree(bound)
        self._beside = _foreign_names(bound.parent)
        self._ancestors = [
            (level, _foreign_names(level))
            for level in bound.parents[1:3]
            if _PYTEST_LEVEL.fullmatch(level.name)
        ]
        self._siblings = _toml_listing(bound.parent, bound)

    def rebase(self) -> None:
        """Take the test's own fixture writes as the new starting point."""
        self._before = _tree(self._bound)

    def check(self) -> None:
        after = _tree(self._bound)
        changed = {p for p in after if self._before.get(p) != after[p]}
        changed |= set(self._before) - set(after)
        stray = sorted(p for p in changed if not self._allowed_path(self._bound / p))
        assert not stray, f"written outside the profile folders: {stray}"
        assert _foreign_names(self._bound.parent) == self._beside, "written beside the test"
        assert real_root_listing() == self._real, "written into the real global root"
        self._before = after

    def check_siblings(self) -> None:
        """No TOML file changed in this worker's other folders; no name appeared above them."""
        assert _toml_listing(self._bound.parent, self._bound) == self._siblings
        for level, names in self._ancestors:
            assert _foreign_names(level) == names, f"written into {level}"

    def _allowed_path(self, path: Path) -> bool:
        resolved = path.resolve()
        return (
            path.name.endswith(".lock")
            or path.name == "state.toml"
            or any(resolved.is_relative_to(folder) for folder in self._allowed)
        )


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    """A fresh world whose working folder is empty, so relative paths name nothing real."""
    world = World(tmp_path / "world")
    enter(world, monkeypatch, cfg.model_copy())
    cwd = world.base / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(analyze_mod, "ocr_language_supported", lambda code: code == "eng")
    return world


@pytest.fixture
def guard(world: World, tmp_path: Path) -> Iterator[_Guard]:
    watch = _Guard(world, tmp_path)
    yield watch
    watch.check_siblings()


@pytest.fixture
def client(world: World) -> Iterator[TestClient[Any]]:
    with http_client() as test_client:
        yield test_client


def _cli_verdict(result: Result, operation: str) -> dict[str, Any] | None:
    """The command's JSON on success, None on a refusal; anything else fails."""
    event(f"cli {operation} exit {result.exit_code}")
    shown = json.loads(result.output)
    if result.exit_code == 0:
        return shown if isinstance(shown, dict) else {"result": shown}
    assert result.exit_code == 1, result.output
    assert isinstance(shown, dict) and isinstance(shown.get("error"), str) and shown["error"]
    return None


def _http_verdict(response: Any, operation: str) -> Any:
    """The body on success, None on a refusal with a reason; a 5xx or a traceback fails."""
    assert TRACEBACK not in response.text, response.text
    event(f"http {operation} {response.status_code}")
    if response.status_code in REFUSED_STATUSES:
        assert response.json().get("detail"), response.text
        return None
    assert response.status_code == 200, (response.status_code, response.text)
    return response.json()


def _mcp_verdict(tool: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
    payload = asyncio.run(mcp_call(tool, arguments))
    assert TRACEBACK not in json.dumps(payload, default=str)
    operation = arguments.get("action", tool)
    event(f"mcp {operation} {'refused' if 'error' in payload else 'ok'}")
    if "error" in payload:
        assert isinstance(payload["error"], str) and payload["error"]
        return None
    return dict(payload)


def _check_written(world: World, name: str, written: dict[str, Any] | None) -> None:
    """A write that succeeded put a file named by the name's slug in the global folder."""
    if written is None:
        return
    path = world.global_profiles() / f"{profile_key(written['name'])}.toml"
    assert path.is_file(), written
    assert written["name"] == name, written


def _cli_safe(value: str) -> bool:
    """An argument an operating system can pass to a process: no NUL character."""
    return "\x00" not in value


def _cli_new(world: World, name: str) -> Result:
    # "--" ends the options, so a name that starts with a dash stays a name
    return cli(world, ["profile", "new", "--target", "global", "--", name], json_mode=True)


@pytest.mark.parametrize("name", ["Plain name", "-dash first"])
def test_the_cli_names_driver_writes_a_valid_name(world: World, name: str) -> None:
    result = _cli_new(world, name)
    assert result.exit_code == 0, result.output
    _check_written(world, name, json.loads(result.output))


@FUZZ
@given(name=_names)
@_examples(name=HOSTILE_NAMES)
def test_new_names_through_the_cli(world: World, guard: _Guard, name: str) -> None:
    if not _cli_safe(name):
        return
    _check_written(world, name, _cli_verdict(_cli_new(world, name), "new"))
    guard.check()


@FUZZ
@given(name=_names)
@_examples(name=HOSTILE_NAMES)
def test_new_names_through_http(
    world: World, guard: _Guard, client: TestClient[Any], name: str
) -> None:
    written = _http_verdict(client.post("/api/profiles/new", json={"name": name}), "new")
    _check_written(world, name, written)
    shown = client.get(f"/api/profiles/{quote(name, safe='')}")
    # a reserved word is a fixed route segment under /api/profiles/, never a profile
    if not (shown.status_code == 405 and profile_key(name) in RESERVED_NAME_KEYS):
        _http_verdict(shown, "show")
    guard.check()


@FUZZ
@given(name=_names)
@_examples(name=HOSTILE_NAMES)
def test_new_names_through_mcp(world: World, guard: _Guard, name: str) -> None:
    _check_written(world, name, _mcp_verdict("profile_manage", {"action": "new", "name": name}))
    guard.check()


def _imported_is_valid(world: World, written: dict[str, Any] | None, valid: bool) -> None:
    """An import that succeeded took a file validate calls valid."""
    if written is None:
        return
    assert valid, written
    assert Path(written["path"]).resolve().is_relative_to(world.global_profiles().resolve())


@FUZZ
@given(raw=_raw_bytes)
@_examples(raw=(*HOSTILE_BYTES, *(text.encode("utf-8") for text in HOSTILE_TEXTS)))
def test_profile_files_through_the_cli(world: World, guard: _Guard, raw: bytes) -> None:
    path = world.base.parent / "fixtures" / "upload.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(raw)
    guard.rebase()
    checked = cli(world, ["profile", "validate", str(path)], json_mode=True)
    assert checked.exit_code in (0, 1), checked.output
    shown = json.loads(checked.output)
    assert shown["valid"] is (checked.exit_code == 0) and isinstance(shown["problems"], list)
    imported = cli(world, ["profile", "import", str(path), "--target", "global"], json_mode=True)
    _imported_is_valid(world, _cli_verdict(imported, "import"), shown["valid"])
    guard.check()


@FUZZ
@given(text=_texts, filename=_filenames)
@_upload_examples
def test_profile_uploads_through_http(
    world: World, guard: _Guard, client: TestClient[Any], text: str, filename: str
) -> None:
    body = {"content": text, "filename": filename}
    shown = _http_verdict(client.post("/api/profiles/validate", json=body), "validate")
    valid = shown is not None and shown["valid"]
    written = _http_verdict(client.post("/api/profiles/import", json=body), "import")
    _imported_is_valid(world, written, valid)
    guard.check()


@FUZZ
@given(text=_texts, filename=_filenames)
@_upload_examples
def test_profile_uploads_through_mcp(world: World, guard: _Guard, text: str, filename: str) -> None:
    args = {"content": text, "filename": filename}
    shown = _mcp_verdict("profile_manage", {"action": "validate", **args})
    valid = shown is not None and shown["valid"]
    written = _mcp_verdict("profile_manage", {"action": "import", **args})
    _imported_is_valid(world, written, valid)
    guard.check()


@pytest.mark.parametrize("route", ["/api/profiles/validate", "/api/profiles/import"])
def test_an_upload_over_the_route_cap_is_refused(
    world: World, guard: _Guard, client: TestClient[Any], route: str
) -> None:
    content = "x" * (PROFILE_BODY_MAX_BYTES + 1)
    response = client.post(route, json={"content": content, "filename": "big.toml"})
    assert response.status_code == 413, response.text
    guard.check()


@pytest.mark.xfail(
    strict=True,
    reason="every JSON route answers 500 when the body is not UTF-8: the decode error "
    "is raised while the request is read, before any handler can refuse it",
)
@pytest.mark.parametrize("route", ["/api/profiles/validate", "/api/profiles/import"])
def test_an_upload_body_that_is_not_utf8_is_refused(
    world: World, guard: _Guard, client: TestClient[Any], route: str
) -> None:
    body = b'{"content": "\xff\xfe", "filename": "x.toml"}'
    assert not _decodes(body)
    response = client.post(route, content=body, headers={"content-type": "application/json"})
    assert response.status_code == 400, response.text
    assert TRACEBACK not in response.text
    guard.check()


def _decodes(body: bytes) -> bool:
    try:
        body.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _seed_folders(world: World) -> None:
    (world.notes / "note.md").write_text(f"# Notiz\n\n{GERMAN}", encoding="utf-8")
    (world.inbox / "plain.txt").write_text("plain", encoding="utf-8")
    link = world.base / "link"
    # Windows without the symlink privilege makes no link; the spec then names nothing
    with contextlib.suppress(OSError):
        if not link.exists():
            link.symlink_to(world.notes, target_is_directory=True)


def _folder(world: World, spec: str) -> str:
    return spec.replace(WORLD, str(world.base))


@FUZZ
@given(spec=_folders)
@_examples(spec=HOSTILE_FOLDERS)
def test_analyze_folders_through_the_cli(world: World, guard: _Guard, spec: str) -> None:
    if not _cli_safe(spec):
        return
    _seed_folders(world)
    guard.rebase()
    folder = _folder(world, spec)
    _cli_verdict(cli(world, ["analyze", "--", folder], json_mode=True), "analyze")
    guard.check()


@FUZZ
@given(spec=_folders)
@_examples(spec=HOSTILE_FOLDERS)
def test_analyze_folders_through_http(
    world: World, guard: _Guard, client: TestClient[Any], spec: str
) -> None:
    _seed_folders(world)
    guard.rebase()
    response = client.post("/api/analyze", json={"directory": _folder(world, spec)})
    assert TRACEBACK not in response.text, response.text
    if response.status_code == 200:
        event("http analyze 200")
        assert "event: done" in response.text, response.text
    else:
        _http_verdict(response, "analyze")
    guard.check()


@FUZZ
@given(spec=_folders)
@_examples(spec=HOSTILE_FOLDERS)
def test_analyze_folders_through_mcp(world: World, guard: _Guard, spec: str) -> None:
    _seed_folders(world)
    guard.rebase()
    _mcp_verdict("analyze", {"directory": _folder(world, spec)})
    guard.check()
