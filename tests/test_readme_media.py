"""The README's videos must always survive the trip back to GIFs for PyPI."""

import re
from pathlib import Path

import pytest
from tools.readme_media import GIF_BASE, VIDEO_BLOCK, to_gifs

README = Path(__file__).resolve().parents[1] / "README.md"
HERO = "what_is_lilbee"
GIF_IMAGE = re.compile(rf"!\[([^\]]*)\]\({re.escape(GIF_BASE)}([a-z0-9_-]+)\.gif\)")


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def test_every_readme_video_becomes_its_own_captioned_gif() -> None:
    source = _readme()
    videos = [(m["demo"], m["caption"]) for m in VIDEO_BLOCK.finditer(source)]
    assert videos, "the README should still show its demos as GitHub video players"

    rendered = to_gifs(source)

    assert "user-attachments/assets" not in rendered
    assert "<!-- demo:" not in rendered
    # Caption and demo must stay paired: a GIF carrying the wrong reel's caption
    # is the failure this rebuild exists to avoid, and counting cannot see it.
    hero, *rebuilt = [(demo, caption) for caption, demo in GIF_IMAGE.findall(rendered)]
    assert hero[0] == HERO, "the hero GIF still leads the page"
    assert rebuilt == videos, "each video becomes its own captioned GIF, in order"


def test_the_hero_is_the_only_gif_left_on_github() -> None:
    demos = [demo for _, demo in GIF_IMAGE.findall(_readme())]
    assert demos == [HERO], "every demo but the hero should be a video on GitHub"


def test_a_video_without_a_caption_comment_is_rejected() -> None:
    orphan = (
        "prose\n\nhttps://github.com/user-attachments/assets/0f3aac4b-51fa-4813-b2b9-000000000000\n"
    )
    with pytest.raises(ValueError, match="no GIF to fall back to"):
        to_gifs(orphan)
