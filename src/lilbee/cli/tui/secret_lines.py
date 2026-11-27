"""The one rule for chat lines that must stay out of the input history."""

from __future__ import annotations

from urllib.parse import urlparse

from lilbee.app.settings_map import SETTINGS_MAP
from lilbee.cli.tui.preset_commands import is_preset_line
from lilbee.crawler.url_filter import is_url

_LOGIN = "/login"
_SET = "/set"
_CRAWL = "/crawl"
_ADD = "/add"
_USERINFO_END = "@"
_ASSIGN = "="


def _has_userinfo(word: str) -> bool:
    """Whether *word* is a URL with a username or password in its host part."""
    if _USERINFO_END not in word:
        return False
    try:
        parsed = urlparse(word)
        return bool(parsed.username or parsed.password)
    except ValueError:
        # A URL that does not parse still holds the "@": keep it out, as the safe answer.
        return True


def _set_word_has_userinfo(word: str) -> bool:
    """Whether a /set word, or the value after its ``=``, is a URL with a password."""
    return _has_userinfo(word) or _has_userinfo(word.partition(_ASSIGN)[2])


def _is_secret_setting(word: str) -> bool:
    """Whether the first /set word, as ``key`` or ``key=value`` in any case, names a secret."""
    key = word.partition(_ASSIGN)[0].lower()
    defn = SETTINGS_MAP.get(key)
    return defn is not None and defn.secret


def carries_secret(text: str) -> bool:
    """Whether the chat line *text* holds a secret, with the dispatcher's word split.

    A /preset line, a /login line with a token, a /set line for a secret setting
    and a /set, /crawl or /add line with a URL that holds a password all count.
    """
    words = text.split()
    command = words[0].lower() if words else ""
    if is_preset_line(text):
        return True
    if command == _LOGIN:
        return len(words) > 1
    if command == _SET and len(words) > 1:
        return _is_secret_setting(words[1]) or any(_set_word_has_userinfo(w) for w in words[1:])
    if command == _CRAWL:
        # The crawl command reads only the first word as the address.
        address = words[1] if len(words) > 1 else ""
        return _has_userinfo(address if is_url(address) else f"https://{address}")
    if command == _ADD:
        return any(is_url(w) and _has_userinfo(w) for w in words[1:])
    return False
