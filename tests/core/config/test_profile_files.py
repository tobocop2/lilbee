"""Profile files: the format, validation, the built-ins, and discovery across folders."""

import re
import tomllib
from pathlib import Path

import pytest

from lilbee.core import profile_files
from lilbee.core.config import cfg
from lilbee.core.config.enums import ProfileScope
from lilbee.core.config.resolve import PROFILE_FIELDS
from lilbee.core.profile_files import (
    BUILTIN_DIRNAME,
    DEFAULT_PROFILE_NAME,
    PACKAGE_PROFILES_DIR,
    PROFILES_DIRNAME,
    ProfileAuthor,
    ProfileFolder,
    ProfileStore,
    profile_folders,
    profile_key,
    read_entry,
    scan,
)

_LANGUAGE_KEYS = {"ocr_language", "fts_language"}


def _write(folder: Path, stem: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _entry(tmp_path: Path, text: str, folder: ProfileFolder = ProfileFolder.GLOBAL):
    return read_entry(_write(tmp_path / "p", "court-filings", text), folder)


def test_values_only_file_is_valid_and_named_by_stem(tmp_path):
    entry = _entry(tmp_path, "[values]\ntable_extraction = true\nchunk_size = 768\n")
    assert entry.error is None
    assert entry.name == "court-filings"
    assert entry.file is not None
    assert dict(entry.file.values) == {"table_extraction": True, "chunk_size": 768}
    assert entry.file.authors == ()
    assert entry.file.format == 1


def test_full_metadata_parses_authors_and_tested_on(tmp_path):
    entry = _entry(
        tmp_path,
        '[profile]\nname = "Court filings"\ndescription = "Scanned court PDFs."\n'
        'authors = [{ name = "Jane Doe", github = "janedoe" }, { name = "Sam" }]\n'
        'tested_on = "4,000 county filings"\nformat = 1\nmin_lilbee = "0.1.0"\n'
        'evidence = "https://example.org/run"\n[values]\nchunk_size = 768\n',
    )
    assert entry.error is None
    assert entry.file is not None
    assert entry.name == entry.file.name == "Court filings"
    assert entry.file.authors == (ProfileAuthor("Jane Doe", "janedoe"), ProfileAuthor("Sam", None))
    assert entry.file.tested_on == "4,000 county filings"
    assert entry.file.description == "Scanned court PDFs."
    assert (entry.file.min_lilbee, entry.file.evidence) == ("0.1.0", "https://example.org/run")


def test_unknown_key_names_min_lilbee_when_newer(tmp_path):
    entry = _entry(tmp_path, '[profile]\nmin_lilbee = "999.0"\n[values]\nbrand_new_knob = 3\n')
    assert entry.file is None
    assert entry.error == "Needs lilbee 999.0 or newer"


def test_bad_value_under_newer_min_lilbee_names_the_version(tmp_path):
    entry = _entry(tmp_path, '[profile]\nmin_lilbee = "999.0"\n[values]\nocr_strategy = "next"\n')
    assert entry.error == "Needs lilbee 999.0 or newer"


def test_unknown_key_without_min_lilbee_is_unknown_setting(tmp_path):
    entry = _entry(tmp_path, '[profile]\nmin_lilbee = "0.1"\n[values]\nbrand_new_knob = 3\n')
    assert entry.error == "Unknown setting: brand_new_knob"
    assert _entry(tmp_path, "[values]\nbrand_new_knob = 3\n").error == (
        "Unknown setting: brand_new_knob"
    )


@pytest.mark.parametrize(
    "key", ["chat_model", "num_ctx", "hf_token", "ocr_timeout", "table_model", "wiki", "theme"]
)
def test_disallowed_key_is_broken_with_reason(tmp_path, key):
    entry = _entry(tmp_path, f'[values]\n{key} = "x"\n')
    assert entry.file is None
    assert entry.error == f"Profiles cannot set {key}"


def test_bad_value_carries_validator_message(tmp_path):
    entry = _entry(tmp_path, "[values]\nchunk_size = 10\n")
    assert entry.error == "Bad value for chunk_size: Input should be greater than or equal to 64"
    bad_enum = _entry(tmp_path, '[values]\nocr_strategy = "sometimes"\n')
    assert bad_enum.error is not None
    assert bad_enum.error.startswith("Bad value for ocr_strategy: Input should be 'auto'")


def test_overlap_not_below_size_in_file_is_broken(tmp_path):
    entry = _entry(tmp_path, "[values]\nchunk_size = 256\nchunk_overlap = 256\n")
    assert entry.error == "chunk_overlap (256) must be < chunk_size (256)"
    assert _entry(tmp_path, "[values]\nchunk_size = 256\nchunk_overlap = 255\n").error is None


@pytest.mark.parametrize("folder", [ProfileFolder.COMMUNITY, ProfileFolder.BUILTIN])
def test_package_retrieval_value_without_evidence_is_broken(tmp_path, folder):
    entry = _entry(tmp_path, "[values]\ntop_k = 20\n", folder)
    assert entry.error == "Sets retrieval setting top_k without evidence"
    same_as_builtin = _entry(tmp_path, "[values]\ntop_k = 12\nchunk_size = 768\n", folder)
    assert same_as_builtin.error is None
    with_evidence = _entry(tmp_path, '[profile]\nevidence = "bb-x"\n[values]\ntop_k = 20\n', folder)
    assert with_evidence.error is None


@pytest.mark.parametrize("folder", [ProfileFolder.PROJECT, ProfileFolder.GLOBAL])
def test_user_retrieval_value_without_evidence_is_valid(tmp_path, folder):
    entry = _entry(tmp_path, "[values]\ntop_k = 20\n", folder)
    assert entry.error is None
    assert entry.file is not None
    assert entry.file.values["top_k"] == 20


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('[profile]\nname = "a/b"\n[values]\n', "Bad name 'a/b'"),
        ('[profile]\nname = ""\n[values]\n', "Bad name ''"),
        ('[profile]\nname = " - _ "\n[values]\n', "Bad name ' - _ '"),
        (f'[profile]\nname = "{"x" * 41}"\n[values]\n', "Bad name"),
        ("[profile]\nformat = 2\n[values]\n", "Unknown profile format 2"),
        ("[profile]\nformat = true\n[values]\n", "Unknown profile format True"),
        ("[profile]\nlicense = 1\n[values]\n", "Unknown profile field: license"),
        ('[profile]\nmin_lilbee = "soon"\n[values]\n', "min_lilbee is not a version: 'soon'"),
        ("[profile]\ndescription = 3\n[values]\n", "description must be text"),
        ("[profile]\nauthors = 3\n[values]\n", "authors must be a list"),
        ('[profile]\nauthors = [{ github = "x" }]\n[values]\n', "Each author needs a name"),
        ('[profile]\nauthors = [{ name = "x", email = "y" }]\n[values]\n', "Unknown author field"),
        ('[profile]\nauthors = [{ name = "x", github = 1 }]\n[values]\n', "github must be text"),
        ('profile = "x"\n[values]\n', "[profile] must be a table"),
        ("[extras]\n[values]\n", "Unknown table: extras"),
        ('[profile]\nname = "x"\n', "Missing [values] table"),
        ("[values\n", "Not valid TOML"),
    ],
)
def test_bad_name_and_unknown_format_are_broken(tmp_path, text, error):
    entry = _entry(tmp_path, text)
    assert entry.file is None
    assert entry.error is not None
    assert entry.error.startswith(error)


def test_broken_file_keeps_its_own_name(tmp_path):
    entry = _entry(tmp_path, '[profile]\nname = "Court filings"\n[values]\nchat_model = "x"\n')
    assert (entry.name, entry.error) == ("Court filings", "Profiles cannot set chat_model")


def test_unreadable_file_is_broken(tmp_path):
    folder = tmp_path / "p" / "dir.toml"
    folder.mkdir(parents=True)
    entry = read_entry(folder, ProfileFolder.GLOBAL)
    assert entry.error is not None
    assert entry.error.startswith("Cannot read the file")


def test_non_utf8_file_is_broken_and_costs_only_itself():
    folder = cfg.data_root / PROFILES_DIRNAME
    _write(folder, "good", "[values]\nchunk_size = 900\n")
    raw = b"# caf\xe9\n[values]\nchunk_size = 800\n"
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    (folder / "latin.toml").write_bytes(raw)
    catalog = ProfileStore().scan()
    latin = catalog.find("latin")
    assert latin is not None and latin.file is None
    assert latin.error == "Not UTF-8 text"
    good = catalog.find("good")
    assert good is not None and good.file is not None


def test_deeply_nested_file_is_broken_and_costs_only_itself():
    folder = cfg.data_root / PROFILES_DIRNAME
    _write(folder, "good", "[values]\nchunk_size = 900\n")
    nested = "x = " + "[" * 1000 + "]" * 1000 + "\n"
    with pytest.raises(RecursionError):
        tomllib.loads(nested)
    _write(folder, "deep", nested)
    catalog = ProfileStore().scan()
    deep = catalog.find("deep")
    assert deep is not None and deep.file is None
    assert deep.error is not None and deep.error.startswith("Not valid TOML: ")
    good = catalog.find("good")
    assert good is not None and good.file is not None


def test_value_whose_validator_raises_a_type_error_is_broken(tmp_path):
    entry = _entry(tmp_path, "[values]\nocr_language = 5\n")
    assert entry.file is None
    assert entry.error is not None
    assert entry.error.startswith("Bad value for ocr_language: ")


_TOML_VALUES = (
    "5",
    "-3",
    "0.5",
    "true",
    '"banana"',
    '""',
    "[]",
    '[1, "a"]',
    "{ a = 1 }",
    "1979-05-27T07:32:00Z",
    "1979-05-27",
    "07:32:00",
)


def test_every_profile_field_with_every_toml_type_is_listed_never_raised():
    folder = cfg.data_root / PROFILES_DIRNAME
    cases = [(key, value) for key in sorted(PROFILE_FIELDS) for value in _TOML_VALUES]
    for index, (key, value) in enumerate(cases):
        _write(folder, f"case-{index:04d}", f"[values]\n{key} = {value}\n")
    _write(folder, "keeper", "[values]\nchunk_size = 900\n")
    project = [e for e in ProfileStore().scan().entries if e.folder is ProfileFolder.PROJECT]
    assert len(cases) > 300
    assert len(project) == len(cases) + 1
    broken = [e for e in project if e.file is None]
    assert all(e.error for e in broken)
    assert all(e.error is None for e in project if e.file is not None)
    assert 0 < len(broken) < len(project)
    keeper = next(e for e in project if e.name == "keeper")
    assert keeper.file is not None


@pytest.mark.parametrize(
    ("running", "min_lilbee", "error"),
    [
        ("0.6.90b447", "0.6.90", "Needs lilbee 0.6.90 or newer"),
        ("0.6.90b447", "0.6.90b1", "Unknown setting: future_key"),
        ("0.6.9", "0.6.10", "Needs lilbee 0.6.10 or newer"),
        ("0.6.10", "0.6.9", "Unknown setting: future_key"),
    ],
)
def test_min_lilbee_compares_as_pep440_versions(tmp_path, monkeypatch, running, min_lilbee, error):
    monkeypatch.setattr(profile_files, "installed_version", lambda _dist: running)
    text = f'[profile]\nmin_lilbee = "{min_lilbee}"\n[values]\nfuture_key = 1\n'
    assert _entry(tmp_path, text).error == error


def _folders(tmp_path: Path) -> list[tuple[ProfileFolder, Path]]:
    return [
        (ProfileFolder.PROJECT, tmp_path / "project"),
        (ProfileFolder.GLOBAL, tmp_path / "global"),
        (ProfileFolder.COMMUNITY, tmp_path / "community"),
        (ProfileFolder.BUILTIN, PACKAGE_PROFILES_DIR / BUILTIN_DIRNAME),
    ]


def test_project_beats_global_beats_community_beats_builtin_and_losers_are_shadowed(tmp_path):
    body = '[profile]\nname = "Court filings"\n[values]\nchunk_size = {}\n'
    _write(tmp_path / "project", "court", body.format(900))
    _write(tmp_path / "global", "court", body.format(800))
    _write(tmp_path / "community", "court", body.format(700))
    _write(tmp_path / "global", "only-global", "[values]\nchunk_size = 600\n")
    catalog = scan(_folders(tmp_path))
    court = [e for e in catalog.entries if e.name == "Court filings"]
    assert [(e.folder, e.shadowed_by) for e in court] == [
        (ProfileFolder.PROJECT, None),
        (ProfileFolder.GLOBAL, ProfileFolder.PROJECT),
        (ProfileFolder.COMMUNITY, ProfileFolder.PROJECT),
    ]
    winner = catalog.find("court filings")
    assert winner is not None and winner.file is not None
    assert winner.file.values["chunk_size"] == 900
    only_global = catalog.find("only-global")
    assert only_global is not None and only_global.folder is ProfileFolder.GLOBAL
    builtin = catalog.find("scanned archive")
    assert builtin is not None and builtin.folder is ProfileFolder.BUILTIN


def test_broken_higher_file_still_takes_the_name(tmp_path):
    _write(tmp_path / "project", "court", '[profile]\nname = "Court"\n[values]\nchat_model = "x"\n')
    _write(tmp_path / "global", "court", '[profile]\nname = "Court"\n[values]\nchunk_size = 800\n')
    winner = scan(_folders(tmp_path)).find("Court")
    assert winner is not None
    assert (winner.folder, winner.error) == (
        ProfileFolder.PROJECT,
        "Profiles cannot set chat_model",
    )


@pytest.mark.parametrize(
    "folder", [ProfileFolder.PROJECT, ProfileFolder.GLOBAL, ProfileFolder.COMMUNITY]
)
def test_builtin_name_in_any_other_folder_is_broken_reserved(tmp_path, folder):
    body = '[profile]\nname = "default"\n[values]\nchunk_size = 900\n'
    _write(tmp_path / folder.value, "mine", body)
    catalog = scan(_folders(tmp_path))
    reserved = [e for e in catalog.entries if e.folder is folder]
    assert len(reserved) == 1
    assert reserved[0].file is None
    assert reserved[0].error == "Reserved name: default is a built-in profile"
    assert reserved[0].shadowed_by is ProfileFolder.BUILTIN
    picked = catalog.find("Default")
    assert picked is not None and picked.folder is ProfileFolder.BUILTIN


def test_case_duplicates_in_one_folder_are_both_broken(tmp_path):
    _write(tmp_path / "global", "a", '[profile]\nname = "Court Filings"\n[values]\n')
    _write(tmp_path / "global", "b", '[profile]\nname = "court_filings"\n[values]\n')
    _write(tmp_path / "global", "c", '[profile]\nname = "Other"\n[values]\n')
    entries = [e for e in scan(_folders(tmp_path)).entries if e.folder is ProfileFolder.GLOBAL]
    assert [(e.name, e.error) for e in entries] == [
        ("Court Filings", "Duplicate name in the global folder"),
        ("court_filings", "Duplicate name in the global folder"),
        ("Other", None),
    ]


def test_lookup_ignores_case_hyphens_and_underscores(tmp_path):
    _write(tmp_path / "global", "x", '[profile]\nname = "Court filings"\n[values]\n')
    catalog = scan(_folders(tmp_path))
    for spelling in ("court-filings", "COURT_FILINGS", "  court   filings ", "Court-_filings"):
        entry = catalog.find(spelling)
        assert entry is not None, spelling
        assert entry.name == "Court filings"
    assert catalog.find("court filing") is None


@pytest.mark.parametrize(
    ("names", "key"),
    [
        (("Court (A)", "Court A", "court-a", "COURT_(A)"), "court-a"),
        (("a_b", "a b", "A-B", " a - _ b "), "a-b"),
        (
            ("Scanned  Archive (city-records)", "Scanned archive city records"),
            "scanned-archive-city-records",
        ),
    ],
)
def test_key_is_the_file_slug_and_spellings_of_one_name_share_it(names, key):
    assert {profile_key(name) for name in names} == {key}
    assert profile_key(key) == key
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", key)


def test_names_sharing_a_slug_are_duplicates_in_a_folder_and_shadowed_across(tmp_path):
    _write(tmp_path / "project", "one", '[profile]\nname = "Court (A)"\n[values]\n')
    _write(tmp_path / "global", "two", '[profile]\nname = "Court A"\n[values]\n')
    _write(tmp_path / "global", "three", '[profile]\nname = "court_a"\n[values]\n')
    catalog = scan(_folders(tmp_path))
    court = [(e.name, e.folder, e.error, e.shadowed_by) for e in catalog.entries[:3]]
    duplicate = "Duplicate name in the global folder"
    assert court == [
        ("Court (A)", ProfileFolder.PROJECT, None, None),
        ("court_a", ProfileFolder.GLOBAL, duplicate, ProfileFolder.PROJECT),
        ("Court A", ProfileFolder.GLOBAL, duplicate, ProfileFolder.PROJECT),
    ]
    picked = catalog.find("court-a")
    assert picked is not None and picked.name == "Court (A)"


def test_rescan_sees_file_added_after_first_scan(tmp_path):
    store = ProfileStore()
    assert store.scan().find("late") is None
    _write(cfg.data_root / PROFILES_DIRNAME, "late", "[values]\nchunk_size = 900\n")
    entry = store.scan().find("late")
    assert entry is not None
    assert entry.folder is ProfileFolder.PROJECT


def test_global_root_has_no_project_folder(tmp_path):
    project = profile_folders(tmp_path / "proj" / ".lilbee")
    assert [f for f, _ in project] == list(ProfileFolder)
    assert project[0][1] == tmp_path / "proj" / ".lilbee" / PROFILES_DIRNAME
    assert project[1][1] == profile_files.default_data_dir() / PROFILES_DIRNAME
    on_global = profile_folders(profile_files.default_data_dir())
    assert [f for f, _ in on_global] == [
        ProfileFolder.GLOBAL,
        ProfileFolder.COMMUNITY,
        ProfileFolder.BUILTIN,
    ]


def test_missing_folders_are_empty(tmp_path):
    folders = [(ProfileFolder.PROJECT, tmp_path / "none"), (ProfileFolder.GLOBAL, tmp_path / "x")]
    assert scan(folders).entries == ()


def test_every_builtin_file_validates_and_holds_no_language_or_retrieval_key():
    catalog = scan([(ProfileFolder.BUILTIN, PACKAGE_PROFILES_DIR / BUILTIN_DIRNAME)])
    names = {e.name: e for e in catalog.entries}
    assert set(names) == {
        "Default",
        "Scanned archive",
        "Research papers",
        "Code repository",
        "Notes and markdown",
    }
    for entry in catalog.entries:
        assert entry.error is None, (entry.name, entry.error)
        assert entry.file is not None
        assert entry.file.description
        assert not set(entry.file.values) & _LANGUAGE_KEYS
        assert all(PROFILE_FIELDS[k] is ProfileScope.INGEST for k in entry.file.values)
        assert entry.file.evidence is None
    default = names[DEFAULT_PROFILE_NAME].file
    assert default is not None and dict(default.values) == {}


def test_frozen_binary_ships_the_package_profiles_folder():
    repo = Path(__file__).resolve().parents[3]
    build = (repo / "tools" / "wheel-build" / "build_lilbee_binary.sh").read_text(encoding="utf-8")
    source = PACKAGE_PROFILES_DIR.relative_to(repo).as_posix()
    shipped = PACKAGE_PROFILES_DIR.relative_to(repo / "src").as_posix()
    assert f"--include-data-dir={source}={shipped}" in build
