"""Session markdown export: front matter, role sections, sources, fences, file names."""

from __future__ import annotations

import os
import sys

import pytest
import yaml
from markdown_it import MarkdownIt

from lilbee.app.session_export import (
    SLUG_MAX_LEN,
    default_export_name,
    session_markdown,
    write_session_markdown,
)
from lilbee.retrieval.query.formatting import source_markdown_link
from lilbee.sessions import MessageRole, Session, SessionMessage, SessionMeta

_ID = "3f2a1b2c-0000-4000-8000-000000000000"


def _meta(title: str = "Brake specs", forked_from: str = "") -> SessionMeta:
    return SessionMeta(
        id=_ID,
        title=title,
        created_at="2026-09-23T10:00:00+00:00",
        updated_at="2026-09-23T10:05:00+00:00",
        model_ref="qwen3:8b",
        scope="both",
        message_count=0,
        forked_from=forked_from,
    )


def _session(*messages: SessionMessage, title: str = "Brake specs", **meta) -> Session:
    return Session(meta=_meta(title, **meta), messages=messages, summary="folded turns")


def _front_matter(markdown: str) -> dict:
    assert markdown.startswith("---\n")
    block, _, _ = markdown.removeprefix("---\n").partition("\n---\n")
    return yaml.safe_load(block)


def _body(markdown: str) -> str:
    return markdown.removeprefix("---\n").partition("\n---\n")[2]


def _user(content: str) -> SessionMessage:
    return SessionMessage(role=MessageRole.USER, content=content)


def _assistant(content: str, sources: tuple[str, ...] = ()) -> SessionMessage:
    return SessionMessage(role=MessageRole.ASSISTANT, content=content, sources=sources)


def test_front_matter_carries_the_session_metadata():
    fields = _front_matter(session_markdown(_session()))
    assert fields == {
        "title": "Brake specs",
        "session": _ID,
        "model": "qwen3:8b",
        "created": "2026-09-23T10:00:00+00:00",
        "updated": "2026-09-23T10:05:00+00:00",
    }


def test_a_fork_names_the_session_it_came_from():
    fields = _front_matter(session_markdown(_session(forked_from="91bc")))
    assert fields["forked_from"] == "91bc"


@pytest.mark.parametrize(
    "title", ["yes", "null", "# lead", 'a: "quoted" \\ title', "Bremsbeläge 制动"]
)
def test_awkward_titles_round_trip_through_the_front_matter(title):
    markdown = session_markdown(_session(title=title))
    assert _front_matter(markdown)["title"] == title


def test_non_ascii_titles_are_written_as_text_not_escapes():
    markdown = session_markdown(_session(title="制动"))
    assert "title: 制动\n" in markdown.partition("\n---\n")[0]


def test_turns_render_as_role_sections_in_order():
    markdown = session_markdown(_session(_user("what torque?"), _assistant("85 Nm.")))
    assert (
        _body(markdown) == "\n# Brake specs\n\n## User\n\nwhat torque?\n\n## Assistant\n\n85 Nm.\n"
    )


def test_the_title_heading_is_one_line_but_the_front_matter_keeps_it_exact():
    markdown = session_markdown(_session(title="two\nlines  here"))
    assert "\n# two lines here\n" in markdown
    assert _front_matter(markdown)["title"] == "two\nlines  here"


def test_the_compaction_summary_is_not_exported():
    assert "folded turns" not in session_markdown(_session(_user("q")))


def test_structured_sources_become_the_numbered_sources_list():
    markdown = session_markdown(_session(_assistant("85 Nm. [1]", sources=("manual.pdf",))))
    assert markdown.endswith(f"85 Nm. [1]\n\nSources:\n\n1. {source_markdown_link('manual.pdf')}\n")


def test_a_reply_that_already_carries_its_sources_list_is_not_given_a_second():
    content = "85 Nm. [1]\n\nSources:\n\n1. manual.pdf"
    markdown = session_markdown(_session(_assistant(content, sources=("manual.pdf",))))
    assert markdown.count("Sources:") == 1


@pytest.mark.parametrize(
    ("content", "closer"),
    [
        ("partial\n```python\nx = 1", "```"),
        ("partial\n~~~~\nx = 1", "~~~~"),
        ("done\n```\nx\n```\nthen\n````md\ny", "````"),
    ],
)
def test_a_cut_off_code_fence_is_closed_before_the_next_section(content, closer):
    markdown = session_markdown(_session(_assistant(content), _user("next question")))
    assert f"{content}\n{closer}\n\n## User\n\nnext question\n" in markdown


def test_a_block_that_is_not_a_fence_is_left_as_written():
    """An open HTML comment also swallows what follows; only fences are closed."""
    content = "<!-- note\nstill a comment"
    markdown = session_markdown(_session(_assistant(content)))
    assert markdown.endswith(f"## Assistant\n\n{content}\n")


@pytest.mark.parametrize(
    "content",
    [
        "```\nx\n```",
        "````\n```\nstill code\n````",
        "```\n~~~\n```",
        "```\n``` not a closer\n```",
    ],
)
def test_balanced_fences_are_left_alone(content):
    markdown = session_markdown(_session(_assistant(content)))
    assert markdown.endswith(f"## Assistant\n\n{content}\n")


def _headings(markdown: str) -> list[str]:
    """The section headings a CommonMark reader sees in the body."""
    tokens = MarkdownIt("commonmark").parse(_body(markdown))
    return [
        tokens[i + 1].content
        for i, token in enumerate(tokens)
        if token.type == "heading_open" and token.tag == "h2"
    ]


@pytest.mark.parametrize(
    "content",
    [
        "steps:\n\n- item\n\n  ```\n  code",
        "1. step\n\n   ```bash\n   cmd",
        "  ```\nindented top-level fence",
        "- item\n  ```\n  done\n  ```\n\n```\ntop-level cut off",
        "1. step\n\n   ```bash\n   cmd\n```\nafter the list",
    ],
    ids=[
        "bullet-item",
        "numbered-item",
        "indented-top-level",
        "closed-item-then-open",
        "closer-outdented-past-the-item",
    ],
)
def test_every_section_heading_survives_a_cut_off_fence(content):
    markdown = session_markdown(_session(_assistant(content), _user("next question")))
    assert _headings(markdown) == ["Assistant", "User"]


def test_the_sources_list_follows_the_closed_fence():
    markdown = session_markdown(_session(_assistant("```\nx = 1", sources=("manual.pdf",))))
    assert "```\nx = 1\n```\n\nSources:\n" in markdown


def test_default_name_is_the_title_slug_and_the_id_prefix():
    assert default_export_name(_meta("Brake specs!")) == "brake-specs-3f2a1b2c.md"


@pytest.mark.parametrize("title", ["制动", "🙂🙂", "!!!"])
def test_a_title_with_nothing_to_slug_falls_back_to_chat(title):
    assert default_export_name(_meta(title)) == "chat-3f2a1b2c.md"


def test_a_long_title_is_capped_without_a_trailing_hyphen():
    name = default_export_name(_meta("a" * (SLUG_MAX_LEN - 1) + " bcdef" * 100))
    stem = name.removesuffix("-3f2a1b2c.md")
    assert len(stem) <= SLUG_MAX_LEN
    assert not stem.endswith("-")
    assert stem == "a" * (SLUG_MAX_LEN - 1)


def test_writes_the_markdown_to_the_given_file(tmp_path):
    session = _session(_user("q"))
    target = write_session_markdown(session, str(tmp_path / "out.md"))
    assert target == (tmp_path / "out.md").resolve()
    assert target.read_text(encoding="utf-8") == session_markdown(session)


def test_a_directory_gets_the_default_name_inside_it(tmp_path):
    target = write_session_markdown(_session(), str(tmp_path))
    assert target == (tmp_path / "brake-specs-3f2a1b2c.md").resolve()
    assert target.is_file()


@pytest.mark.parametrize("separator", ["/", os.sep])
def test_a_trailing_separator_names_a_directory_to_create(tmp_path, separator):
    target = write_session_markdown(_session(), f"{tmp_path / 'new'}{separator}")
    assert target == (tmp_path / "new" / "brake-specs-3f2a1b2c.md").resolve()
    assert target.is_file()


def test_an_explicit_path_replaces_an_existing_file(tmp_path):
    existing = tmp_path / "out.md"
    existing.write_text("old", encoding="utf-8")
    write_session_markdown(_session(), str(existing))
    assert existing.read_text(encoding="utf-8").startswith("---\n")


def test_a_home_relative_path_expands(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    target = write_session_markdown(_session(), "~/out.md")
    assert target == (tmp_path / "out.md").resolve()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no POSIX mode bits")
def test_the_file_is_owner_only(tmp_path):
    target = write_session_markdown(_session(), str(tmp_path / "out.md"))
    assert os.stat(target).st_mode & 0o777 == 0o600
