"""Tests for the non-destructive config deep-merge."""

from __future__ import annotations

from lilbee.cli.agent_configs.merge import LILBEE_PROVIDER_KEY, deep_merge, prune_lilbee


def test_adds_lilbee_into_empty_config():
    out = deep_merge({}, {"provider": {LILBEE_PROVIDER_KEY: {"name": "lilbee"}}})
    assert out == {"provider": {LILBEE_PROVIDER_KEY: {"name": "lilbee"}}}


def test_preserves_sibling_providers():
    base = {"provider": {"anthropic": {"name": "anthropic"}}, "theme": "dark"}
    deep_merge(base, {"provider": {LILBEE_PROVIDER_KEY: {"name": "lilbee"}}, "model": "lilbee/x"})
    assert base["provider"]["anthropic"] == {"name": "anthropic"}
    assert base["provider"][LILBEE_PROVIDER_KEY] == {"name": "lilbee"}
    assert base["theme"] == "dark"
    assert base["model"] == "lilbee/x"


def test_refresh_overwrites_only_lilbee_leaf():
    base = {"provider": {LILBEE_PROVIDER_KEY: {"options": {"baseURL": "old"}}}}
    deep_merge(base, {"provider": {LILBEE_PROVIDER_KEY: {"options": {"baseURL": "new"}}}})
    assert base["provider"][LILBEE_PROVIDER_KEY]["options"]["baseURL"] == "new"


def test_idempotent():
    frag = {"provider": {LILBEE_PROVIDER_KEY: {"name": "lilbee"}}, "model": "lilbee/x"}
    a = deep_merge({}, frag)
    b = deep_merge(deep_merge({}, frag), frag)
    assert a == b


def test_fragment_leaf_replaces_non_dict_base():
    base = {"model": {"was": "dict"}}
    deep_merge(base, {"model": "lilbee/x"})
    assert base["model"] == "lilbee/x"


def test_prune_removes_only_lilbee_entry():
    base = {"mcp": {"lilbee": {"url": "x"}, "other": {"url": "y"}}}
    prune_lilbee(base, "mcp")
    assert base["mcp"] == {"other": {"url": "y"}}


def test_prune_noop_when_absent():
    base = {"mcp": {"other": {"url": "y"}}}
    prune_lilbee(base, "mcp")
    assert base["mcp"] == {"other": {"url": "y"}}
    prune_lilbee({}, "mcp")  # missing container is a no-op
