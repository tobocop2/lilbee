"""The crawl login the settings store: who gets it, and what the settings refuse."""

from __future__ import annotations

import sys

import pytest

from lilbee.app.settings import apply_settings_update
from lilbee.app.settings_map import SETTINGS_MAP
from lilbee.config_meta import PUBLIC_CONFIG_FIELDS, WRITABLE_CONFIG_FIELDS
from lilbee.core.config import Config, cfg
from lilbee.crawler.models import NO_LOGIN, CrawlLogin
from lilbee.crawler.stored_login import (
    crawl_login,
    login_host,
    stored_login_withheld,
    url_hosts,
)

SECRET_FIELDS = ("crawl_authorization", "crawl_password", "crawl_cookie")
LOGIN_FIELDS = ("crawl_login_host", "crawl_username", *SECRET_FIELDS)


@pytest.fixture
def stored(monkeypatch):
    """A stored cookie and password login for wiki.example.com."""
    monkeypatch.setattr(cfg, "crawl_login_host", "wiki.example.com")
    monkeypatch.setattr(cfg, "crawl_username", "alice")
    monkeypatch.setattr(cfg, "crawl_password", "pw-stored")
    monkeypatch.setattr(cfg, "crawl_cookie", "sid=stored")


@pytest.fixture
def persisted(monkeypatch):
    """Settings writes reach the isolated config.toml, and each batch is also recorded."""
    from lilbee.app import settings as appset

    writes: list[dict] = []
    write_to_file = appset.persistent_settings.update_values

    def record_and_write(root, values):
        writes.append(values)
        write_to_file(root, values)

    monkeypatch.setattr(appset.persistent_settings, "update_values", record_and_write)
    return writes


class TestWhoGetsTheStoredLogin:
    def test_the_host_it_names_gets_it(self, stored):
        login = crawl_login(["https://wiki.example.com/a"])
        assert login == CrawlLogin(username="alice", password="pw-stored", cookie="sid=stored")

    def test_the_match_ignores_scheme_port_and_case(self, stored):
        urls = ["http://WIKI.example.com:8080/a", "https://wiki.example.com/b"]
        assert crawl_login(urls).cookie == "sid=stored"

    @pytest.mark.parametrize(
        "urls",
        [
            ["https://other.example/"],
            ["https://wiki.example.com/", "https://other.example/"],
            ["https://wiki.example.com:99999/"],
        ],
        ids=["another-host", "several-hosts", "unparsed"],
    )
    def test_any_other_crawl_gets_none(self, stored, urls):
        assert crawl_login(urls) == NO_LOGIN
        assert stored_login_withheld(urls)

    def test_no_host_means_no_login(self, stored, monkeypatch):
        monkeypatch.setattr(cfg, "crawl_login_host", "")
        assert crawl_login(["https://wiki.example.com/"]) == NO_LOGIN

    def test_nothing_stored_withholds_nothing(self):
        assert not stored_login_withheld(["https://wiki.example.com/"])

    def test_a_typed_field_wins_over_the_stored_field(self, stored):
        login = crawl_login(["https://wiki.example.com/"], CrawlLogin(cookie="sid=typed"))
        assert login == CrawlLogin(username="alice", password="pw-stored", cookie="sid=typed")

    def test_a_typed_login_for_several_hosts_is_refused(self):
        with pytest.raises(ValueError, match="one host"):
            crawl_login(["https://a.example/", "https://b.example/"], CrawlLogin(cookie="x"))

    def test_a_typed_username_next_to_a_stored_header_is_refused_without_the_value(
        self, monkeypatch
    ):
        monkeypatch.setattr(cfg, "crawl_login_host", "wiki.example.com")
        monkeypatch.setattr(cfg, "crawl_authorization", "Bearer stored-Hv3")
        with pytest.raises(ValueError, match="not both") as caught:
            crawl_login(["https://wiki.example.com/"], CrawlLogin(username="alice"))
        assert "stored-Hv3" not in str(caught.value)

    def test_helpers_read_hosts_as_the_url_standard_does(self):
        assert url_hosts(["https://A.example:81/x"]) == {"a.example"}
        assert login_host("BÜCHER.example") == "xn--bcher-kva.example"
        assert login_host("private example") is None


class TestEnvironmentNamesStillWork:
    @pytest.mark.parametrize(
        ("variable", "field"),
        [
            ("LILBEE_CRAWL_LOGIN_HOST", "crawl_login_host"),
            ("LILBEE_CRAWL_AUTHORIZATION", "crawl_authorization"),
            ("LILBEE_CRAWL_USERNAME", "crawl_username"),
            ("LILBEE_CRAWL_PASSWORD", "crawl_password"),
            ("LILBEE_CRAWL_COOKIE", "crawl_cookie"),
        ],
    )
    def test_the_variable_sets_the_setting(self, monkeypatch, variable, field):
        monkeypatch.setenv(variable, "value-1")
        assert getattr(Config(), field) == "value-1"


class TestSecretsAreHandledLikeApiKeys:
    @pytest.mark.parametrize("field", SECRET_FIELDS)
    def test_a_secret_is_masked_write_only_and_never_public(self, field):
        assert SETTINGS_MAP[field].secret
        assert field in WRITABLE_CONFIG_FIELDS
        assert field not in PUBLIC_CONFIG_FIELDS

    @pytest.mark.parametrize("field", ("crawl_login_host", "crawl_username"))
    def test_the_host_and_the_username_are_plain_settings(self, field):
        assert not SETTINGS_MAP[field].secret
        assert field in PUBLIC_CONFIG_FIELDS

    def test_get_config_leaves_the_secrets_out(self, monkeypatch):
        import asyncio

        from lilbee.server.handlers.config import get_config

        monkeypatch.setattr(cfg, "crawl_cookie", "sid=Ck-7Hn")
        monkeypatch.setattr(cfg, "crawl_username", "alice")
        body = asyncio.run(get_config()).model_dump()
        assert "Ck-7Hn" not in str(body)
        assert not set(SECRET_FIELDS) & body.keys()
        assert body["crawl_username"] == "alice"


class TestSettingsRefuseABadLogin:
    @pytest.mark.parametrize(
        ("updates", "words"),
        [
            ({"crawl_password": "pw-8Lm"}, "password needs a username"),
            ({"crawl_authorization": "Bearer t-8Lm", "crawl_username": "a"}, "not both"),
            ({"crawl_cookie": "sid=8\nLm"}, "cookie holds a control character"),
            ({"crawl_login_host": "private example"}, "must be a host name"),
            ({"crawl_login_host": "https://wiki.example.com"}, "must be a host name"),
            ({"crawl_login_host": "wiki.example.com/wiki"}, "must be a host name"),
            ({"crawl_login_host": "alice@wiki.example.com"}, "must be a host name"),
        ],
        ids=[
            "password-alone",
            "both",
            "control-character",
            "bad-host",
            "url-with-scheme",
            "host-with-path",
            "host-with-userinfo",
        ],
    )
    def test_a_refused_login_names_the_field_and_not_the_value(self, persisted, updates, words):
        with pytest.raises(ValueError, match=words) as caught:
            apply_settings_update(updates)
        assert "8Lm" not in str(caught.value)
        assert all(getattr(cfg, key) == "" for key in updates)

    @pytest.mark.parametrize("field", LOGIN_FIELDS)
    @pytest.mark.parametrize("value", [123456, ["a"], {"k": "v"}], ids=["int", "list", "dict"])
    def test_a_value_that_is_not_text_is_a_validation_error(self, persisted, field, value):
        """Like any other string setting: a ValueError (400 over REST), never a TypeError."""
        with pytest.raises(ValueError):
            apply_settings_update({field: value})
        assert getattr(cfg, field) == ""

    @pytest.mark.parametrize("field", ["crawl_authorization", "crawl_password", "crawl_cookie"])
    @pytest.mark.parametrize("value", [123456, ["leak-6Qd"], {"k": "leak-6Qd"}])
    def test_the_refusal_of_a_secret_does_not_repeat_the_input(self, persisted, field, value):
        with pytest.raises(ValueError) as caught:
            apply_settings_update({field: value})
        assert "leak-6Qd" not in str(caught.value)
        assert "123456" not in str(caught.value)
        assert field in str(caught.value)

    def test_the_refusal_of_a_plain_setting_keeps_the_input(self, persisted):
        with pytest.raises(ValueError, match="leak-6Qd"):
            apply_settings_update({"crawl_username": ["leak-6Qd"]})

    def test_without_ada_url_the_host_check_lets_the_value_through(self, monkeypatch):
        """No crawl runs without ada-url, so config load must not stop on it."""
        from lilbee.core.config.login_host import refuse_unless_host

        monkeypatch.setitem(sys.modules, "ada_url", None)
        refuse_unless_host("https://wiki.example.com/")

    @pytest.mark.parametrize("host", ["wiki.example.com", "WIKI.example.com:8443", "[::1]:80"])
    def test_a_bare_host_with_or_without_a_port_is_stored(self, persisted, host):
        apply_settings_update({"crawl_login_host": host})
        assert cfg.crawl_login_host == host

    def test_a_login_set_one_field_at_a_time_is_checked_against_the_stored_fields(self, persisted):
        apply_settings_update({"crawl_username": "alice"})
        apply_settings_update({"crawl_password": "pw-1"})
        assert (cfg.crawl_username, cfg.crawl_password) == ("alice", "pw-1")
        with pytest.raises(ValueError, match="needs a username"):
            apply_settings_update({"crawl_username": ""})

    def test_a_valid_login_is_stored_and_kept_out_of_the_logs(self, persisted, caplog):
        caplog.set_level("DEBUG")
        apply_settings_update(
            {"crawl_login_host": "wiki.example.com", "crawl_cookie": "sid=Ck-2Pw"}
        )
        assert cfg.crawl_cookie == "sid=Ck-2Pw"
        assert persisted == [{"crawl_login_host": "wiki.example.com", "crawl_cookie": "sid=Ck-2Pw"}]
        assert "Ck-2Pw" not in caplog.text
