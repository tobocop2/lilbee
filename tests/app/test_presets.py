"""The server preset service: its URL check, its store and what it shows."""

import dataclasses
import json
import logging
import re
import socket
import threading

import pytest

from lilbee.app import presets
from lilbee.app import settings as app_settings
from lilbee.config_meta import SERVICE_ONLY_FIELDS, WRITABLE_CONFIG_FIELDS
from lilbee.core import settings
from lilbee.core.config import Config, cfg
from lilbee.providers.roles import WorkerRole
from lilbee.server.handlers.config import get_config, get_config_defaults

KEY = "SENTINEL-key-5d1e"
PUBLIC_HOST = "pub.example.test"
URL = f"http://{PUBLIC_HOST}:8080/v1"
OTHER_URL = "http://10.0.0.7:9000/v1"
BLOCKED_URL = "http://169.254.169.254/"
METADATA_REFUSAL = re.escape("blocked address 169.254.169.254")


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.server_presets = None
    yield
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def dns(monkeypatch):
    """The DNS the service sees: a host maps to its answers, any other host has none."""
    table = {PUBLIC_HOST: ["93.184.216.34"]}

    def getaddrinfo(host, _port):
        if host not in table:
            raise socket.gaierror("no answer")
        return [(2, 1, 6, "", (address, 0)) for address in table[host]]

    monkeypatch.setattr(presets.socket, "getaddrinfo", getaddrinfo)
    return table


def _stored():
    raw = settings.load(cfg.data_root).get("server_presets")
    return json.loads(raw) if raw else []


def _store(*entries):
    settings.set_value(cfg.data_root, "server_presets", json.dumps(list(entries)))


def _entry(name, url=URL, **extra):
    return {"name": name, "url": url, "roles": ["chat"], **extra}


class TestValidatePresetUrl:
    @pytest.mark.parametrize(
        "url",
        [
            URL,
            "https://8.8.8.8/v1",
            "HTTP://8.8.8.8/",
            "http://127.0.0.1:8080",
            "http://127.1/",
            "http://2130706433/",
            "http://0x7f.1/",
            "http://0177.0.0.1/",
            "http://8.8.8.8./",
            "http://10.0.0.1/",
            "http://192.168.1.1/",
            "http://100.64.0.1/",
            "http://[::1]:8080/",
            "http://[fc00::1]/",
            "http://[fec0::1]/",
            "http://192.0.2.1/",
            "http://localhost.nothing.test/",
        ],
    )
    def test_a_url_a_local_or_public_server_uses_is_allowed(self, url):
        assert presets.validate_preset_url(url) is None

    @pytest.mark.parametrize(
        ("url", "blocked"),
        [
            (BLOCKED_URL, "169.254.169.254"),
            ("http://2852039166/", "169.254.169.254"),
            ("http://0xa9fea9fe/", "169.254.169.254"),
            ("http://0xA9.254.0xa9fe/", "169.254.169.254"),
            ("http://0251.0376.0251.0376/", "169.254.169.254"),
            ("http://169.254.43518/", "169.254.169.254"),
            ("http://169.16689662/", "169.254.169.254"),
            ("http://169.254.169.254./", "169.254.169.254"),
            ("http://0/", "0.0.0.0"),
            ("http://0x/", "0.0.0.0"),
            ("http://224.0.0.1/", "224.0.0.1"),
            ("http://240.0.0.1/", "240.0.0.1"),
            ("http://[fe80::1]/", "fe80::1"),
            ("http://[::]/", "::"),
            ("http://[ff02::1]/", "ff02::1"),
            ("http://[64:ff9b::808:808]/", "64:ff9b::808:808"),
            ("http://[::ffff:169.254.169.254]/", "169.254.169.254"),
            ("http://[2002:a9fe:a9fe::]/", "169.254.169.254"),
        ],
    )
    def test_a_blocked_address_is_refused_in_every_form(self, url, blocked):
        with pytest.raises(ValueError, match=re.escape(f"blocked address {blocked}")):
            presets.validate_preset_url(url)

    @pytest.mark.parametrize(
        "host",
        [
            "1.2.3.4.5",
            "08.1.1.1",
            "256.1.1.1",
            "1.2.3.256",
            "1.2.65536",
            "1.16777216",
            "4294967296",
            "foo.1",
            "1..2",
            "0xg.1",
        ],
    )
    def test_a_host_that_ends_in_a_number_and_names_no_address_is_refused(self, host):
        with pytest.raises(ValueError, match="not a valid IPv4 address"):
            presets.validate_preset_url(f"http://{host}/")

    @pytest.mark.parametrize("url", ["http://[v1.abc]/", "http://[v1.a:b]/"])
    def test_a_bracketed_host_that_is_no_ipv6_is_refused(self, url):
        with pytest.raises(ValueError, match="not a valid IPv6 address"):
            presets.validate_preset_url(url)

    @pytest.mark.parametrize(
        ("url", "message"),
        [
            ("ftp://8.8.8.8/", "must start with http"),
            ("8.8.8.8", "must start with http"),
            ("", "printable ASCII"),
            ("http://8.8.8.8/a b", "printable ASCII"),
            ("http://8.8.8.8\\@10.0.0.1/", "printable ASCII"),
            ("http://8.8.8.8/\n", "printable ASCII"),
            ("http://１.１.１.１/", "printable ASCII"),  # noqa: RUF001
            ("http://8.8.8.8/?a=b", "query or a fragment"),
            ("http://8.8.8.8/#a", "query or a fragment"),
            ("http://user:pass@8.8.8.8/", "user name or password"),
            ("http://@8.8.8.8/", "user name or password"),
            ("http://[fe80::1%25en0]/", "percent sign"),
            ("http://%31.1.1.1/", "percent sign"),
            ("http:///v1", "needs a host"),
            ("http://8.8.8.8:99999/", "does not parse"),
            ("http://8.8.8.8:port/", "does not parse"),
            ("http://[::g]/", "does not parse"),
        ],
    )
    def test_a_malformed_url_is_refused(self, url, message):
        with pytest.raises(ValueError, match=message):
            presets.validate_preset_url(url)

    def test_a_refusal_repeats_no_part_of_the_url(self):
        with pytest.raises(ValueError) as refused:
            presets.validate_preset_url(f"http://user:{KEY}@8.8.8.8/")
        assert KEY not in str(refused.value)

    def test_one_blocked_answer_among_good_ones_refuses(self, dns):
        dns["mixed.test"] = ["8.8.8.8", "169.254.169.254"]
        with pytest.raises(ValueError, match=METADATA_REFUSAL):
            presets.validate_preset_url("http://mixed.test/")

    def test_a_host_name_with_a_mapped_answer_is_read_as_the_address_it_embeds(self, dns):
        dns["mapped.test"] = ["::ffff:169.254.169.254"]
        with pytest.raises(ValueError, match=METADATA_REFUSAL):
            presets.validate_preset_url("http://mapped.test/")

    def test_a_host_with_no_dns_answer_is_allowed_at_save(self):
        assert presets.validate_preset_url("http://offline.test/") is None

    def test_a_host_name_dns_cannot_encode_is_refused(self, monkeypatch):
        def refuses(_host, _port):
            raise UnicodeError("label too long")

        monkeypatch.setattr(presets.socket, "getaddrinfo", refuses)
        with pytest.raises(ValueError, match="not a valid host name"):
            presets.validate_preset_url("http://long.test/")


class TestCheckRequestAddress:
    def test_a_public_answer_is_allowed(self):
        assert presets.check_request_address(URL) is None

    def test_a_literal_address_needs_no_dns(self):
        assert presets.check_request_address(OTHER_URL) is None

    def test_a_blocked_literal_is_refused(self):
        with pytest.raises(ValueError, match=METADATA_REFUSAL):
            presets.check_request_address(BLOCKED_URL)

    def test_no_dns_answer_is_refused(self):
        with pytest.raises(ValueError, match="no DNS answer"):
            presets.check_request_address("http://offline.test/")

    def test_an_answer_that_turned_blocked_after_save_is_refused(self, dns):
        presets.add_preset("alpha", URL, roles=["chat"])
        dns[PUBLIC_HOST] = ["169.254.169.254"]
        with pytest.raises(ValueError, match=METADATA_REFUSAL):
            presets.check_request_address(URL)


class TestAdd:
    def test_it_stores_one_json_scalar_with_the_key(self):
        view = presets.add_preset("alpha", URL, roles=["chat", "embed"], api_key=KEY)
        assert view == presets.ServerPreset("alpha", URL, (WorkerRole.CHAT, WorkerRole.EMBED), True)
        assert isinstance(settings.load(cfg.data_root)["server_presets"], str)
        assert _stored() == [
            {"name": "alpha", "url": URL, "roles": ["chat", "embed"], "api_key": KEY}
        ]
        assert json.loads(cfg.server_presets) == _stored()

    @pytest.mark.parametrize("empty", [None, ""])
    def test_an_empty_key_stores_no_key(self, empty):
        view = presets.add_preset("alpha", URL, roles=["chat"], api_key=empty)
        assert view.has_key is False
        assert "api_key" not in _stored()[0]

    def test_a_duplicate_name_is_refused(self):
        presets.add_preset("alpha", URL, roles=["chat"])
        with pytest.raises(ValueError, match="'alpha' already exists"):
            presets.add_preset("alpha", OTHER_URL, roles=["chat"])
        assert [entry["url"] for entry in _stored()] == [URL]

    @pytest.mark.parametrize(
        "name", ["", "a/b", "a b", "Alpha", "-x", "_x", "x.y", "x:y", "naïve", "a\nb"]
    )
    def test_a_name_outside_the_pattern_is_refused(self, name):
        with pytest.raises(ValueError, match=r"^A preset name uses lower-case letters"):
            presets.add_preset(name, URL, roles=["chat"])

    @pytest.mark.parametrize("name", ["openai", "local", "anthropic"])
    def test_a_provider_prefix_is_refused_as_a_name(self, name):
        with pytest.raises(ValueError, match="provider prefix"):
            presets.add_preset(name, URL, roles=["chat"])

    @pytest.mark.parametrize("name", ["x", "9", "9-ok_name", "openai-compat", "local2"])
    def test_a_name_inside_the_pattern_is_allowed(self, name):
        assert presets.add_preset(name, URL, roles=["chat"]).name == name

    def test_no_role_is_refused(self):
        with pytest.raises(ValueError, match="at least one role"):
            presets.add_preset("alpha", URL, roles=[])

    def test_an_unknown_role_is_refused_by_its_name(self):
        with pytest.raises(ValueError, match="Unknown role 'speak'"):
            presets.add_preset("alpha", URL, roles=["chat", "speak"])

    def test_a_repeated_role_is_refused(self):
        with pytest.raises(ValueError, match="each role once"):
            presets.add_preset("alpha", URL, roles=["chat", "chat"])

    def test_a_blocked_url_writes_nothing(self):
        with pytest.raises(ValueError, match="blocked address"):
            presets.add_preset("alpha", BLOCKED_URL, roles=["chat"])
        assert _stored() == []

    def test_the_service_ignores_the_http_flag(self):
        cfg.allow_http_engines = False
        assert presets.add_preset("off", URL, roles=["chat"]).name == "off"
        cfg.allow_http_engines = True
        assert presets.add_preset("on", URL, roles=["chat"]).name == "on"
        with pytest.raises(ValueError, match="blocked address"):
            presets.add_preset("x", BLOCKED_URL, roles=["chat"])

    def test_an_entry_another_process_wrote_survives(self):
        _store(_entry("from-cli", OTHER_URL))
        presets.add_preset("from-server", URL, roles=["chat"])
        assert [entry["name"] for entry in _stored()] == ["from-cli", "from-server"]

    def test_concurrent_adds_all_survive(self):
        threads = [
            threading.Thread(
                target=presets.add_preset, args=(f"t{i}", URL), kwargs={"roles": ["chat"]}
            )
            for i in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(_stored()) == 8


class TestEdit:
    @pytest.fixture(autouse=True)
    def _alpha(self):
        presets.add_preset("alpha", URL, roles=["chat"], api_key=KEY)

    def test_a_url_edit_changes_the_url_and_keeps_the_rest(self):
        view = presets.edit_preset("alpha", url=OTHER_URL)
        assert view.base_url == OTHER_URL
        assert _stored() == [{"name": "alpha", "url": OTHER_URL, "roles": ["chat"], "api_key": KEY}]

    def test_a_role_edit_changes_the_roles_and_keeps_the_rest(self):
        view = presets.edit_preset("alpha", roles=["rerank"])
        assert view.roles == (WorkerRole.RERANK,)
        assert _stored() == [{"name": "alpha", "url": URL, "roles": ["rerank"], "api_key": KEY}]

    @pytest.mark.parametrize("empty", [None, ""])
    def test_an_empty_or_omitted_key_keeps_the_stored_key(self, empty):
        view = presets.edit_preset("alpha", roles=["vision"], api_key=empty)
        assert view.has_key is True
        assert _stored()[0]["api_key"] == KEY

    def test_a_new_key_replaces_the_old_key(self):
        presets.edit_preset("alpha", api_key=KEY + "-2")
        assert _stored()[0]["api_key"] == KEY + "-2"

    def test_a_blocked_url_leaves_the_entry(self):
        with pytest.raises(ValueError, match="blocked address"):
            presets.edit_preset("alpha", url=BLOCKED_URL)
        assert _stored()[0]["url"] == URL

    def test_an_empty_role_list_is_refused(self):
        with pytest.raises(ValueError, match="at least one role"):
            presets.edit_preset("alpha", roles=[])

    def test_an_unknown_name_is_refused(self):
        with pytest.raises(ValueError, match="No server preset is named 'nobody'"):
            presets.edit_preset("nobody", url=URL)


class TestClearKey:
    def test_the_key_leaves_the_file_the_config_and_every_read(self):
        presets.add_preset("alpha", URL, roles=["chat"], api_key=KEY)
        view = presets.clear_key("alpha")
        assert view.has_key is False
        assert KEY not in (cfg.data_root / "config.toml").read_text(encoding="utf-8")
        assert KEY not in cfg.server_presets
        assert [entry.has_key for entry in presets.list_presets()] == [False]
        assert _stored() == [{"name": "alpha", "url": URL, "roles": ["chat"]}]

    def test_it_leaves_the_other_presets_alone(self):
        presets.add_preset("alpha", URL, roles=["chat"], api_key=KEY)
        presets.add_preset("beta", OTHER_URL, roles=["embed"], api_key=KEY + "-beta")
        presets.clear_key("alpha")
        assert [entry.has_key for entry in presets.list_presets()] == [False, True]

    def test_an_unknown_name_is_refused(self):
        with pytest.raises(ValueError, match="No server preset is named 'nobody'"):
            presets.clear_key("nobody")


class TestRemove:
    def test_the_last_remove_leaves_no_field_in_the_file(self):
        presets.add_preset("alpha", URL, roles=["chat"], api_key=KEY)
        presets.remove_preset("alpha")
        assert "server_presets" not in settings.load(cfg.data_root)
        assert cfg.server_presets is None
        assert presets.list_presets() == []

    def test_it_leaves_the_other_presets_alone(self):
        presets.add_preset("alpha", URL, roles=["chat"])
        presets.add_preset("beta", OTHER_URL, roles=["chat"])
        presets.remove_preset("alpha")
        assert [entry.name for entry in presets.list_presets()] == ["beta"]

    def test_an_unknown_name_is_refused(self):
        with pytest.raises(ValueError, match="No server preset is named 'nobody'"):
            presets.remove_preset("nobody")

    def test_each_of_two_blocked_entries_can_be_removed(self):
        _store(_entry("bad-one", BLOCKED_URL), _entry("bad-two", "http://224.0.0.1/"))
        presets.remove_preset("bad-one")
        assert [entry["name"] for entry in _stored()] == ["bad-two"]
        presets.remove_preset("bad-two")
        assert _stored() == []


class TestWholeList:
    @pytest.mark.parametrize(
        ("entries", "message"),
        [
            ([_entry("bad", BLOCKED_URL)], "'bad' is refused.*blocked address"),
            ([_entry("Bad Name")], "'Bad Name' is refused.*lower-case letters"),
            ([_entry("openai")], "'openai' is refused.*provider prefix"),
            ([_entry("twin"), _entry("twin")], "share a name"),
        ],
    )
    def test_a_bad_stored_list_stops_a_write_and_the_file_stays(self, entries, message):
        _store(*entries)
        before = (cfg.data_root / "config.toml").read_bytes()
        with pytest.raises(ValueError, match=message):
            presets.add_preset("good", URL, roles=["chat"])
        assert (cfg.data_root / "config.toml").read_bytes() == before

    def test_a_stored_blocked_entry_can_be_edited_to_a_good_url(self):
        _store(_entry("bad", BLOCKED_URL))
        presets.edit_preset("bad", url=URL)
        assert _stored()[0]["url"] == URL


class TestDnsAndTheLock:
    def test_the_new_url_is_looked_up_with_the_config_lock_free(self, monkeypatch):
        held = []

        def getaddrinfo(_host, _port):
            held.append(settings._settings_lock.locked())
            return [(2, 1, 6, "", ("8.8.8.8", 0))]

        monkeypatch.setattr(presets.socket, "getaddrinfo", getaddrinfo)
        presets.add_preset("alpha", URL, roles=["chat"])
        presets.edit_preset("alpha", url="http://other.test/")
        assert held == [False, False]

    def test_a_write_looks_up_no_stored_host_name(self, monkeypatch):
        presets.add_preset("alpha", URL, roles=["chat"])
        asked = []

        def getaddrinfo(host, _port):
            asked.append(host)
            raise socket.gaierror("no answer")

        monkeypatch.setattr(presets.socket, "getaddrinfo", getaddrinfo)
        presets.add_preset("beta", OTHER_URL, roles=["chat"])
        presets.edit_preset("alpha", roles=["embed"])
        presets.clear_key("alpha")
        presets.remove_preset("beta")
        assert asked == []

    def test_a_stored_name_whose_answer_turned_blocked_does_not_stop_a_write(self, dns):
        presets.add_preset("alpha", URL, roles=["chat"])
        dns[PUBLIC_HOST] = ["169.254.169.254"]
        presets.add_preset("beta", OTHER_URL, roles=["chat"])
        assert [entry["name"] for entry in _stored()] == ["alpha", "beta"]
        with pytest.raises(ValueError, match="blocked address"):
            presets.check_request_address(URL)


class TestList:
    def test_an_absent_store_is_an_empty_list(self):
        assert presets.list_presets() == []

    def test_an_entry_says_a_key_is_set_and_holds_no_key(self):
        presets.add_preset("alpha", URL, roles=["chat"], api_key=KEY)
        view = presets.list_presets()[0]
        assert view == presets.ServerPreset("alpha", URL, (WorkerRole.CHAT,), True)
        assert not hasattr(view, "api_key")
        assert KEY not in f"{view!r} {dataclasses.asdict(view)}"

    @pytest.mark.parametrize(
        "raw",
        [
            f"not json {KEY}",
            json.dumps({"name": "alpha", "api_key": KEY}),
            json.dumps([{"name": "alpha", "url": URL, "roles": [], "api_key": KEY}]),
            json.dumps([{"name": "alpha", "url": URL, "roles": ["speak"], "api_key": KEY}]),
            json.dumps([{"name": "alpha", "url": URL, "roles": ["chat"], "extra": KEY}]),
            [{"api_key": KEY}],
            7,
        ],
    )
    def test_a_stored_value_that_does_not_parse_is_refused_and_logged_with_no_key(
        self, raw, caplog
    ):
        settings.set_value(cfg.data_root, "server_presets", raw)
        with (
            caplog.at_level(logging.WARNING, logger=presets.__name__),
            pytest.raises(ValueError, match="do not parse") as refused,
        ):
            presets.list_presets()
        assert "do not parse" in caplog.text
        assert KEY not in caplog.text
        assert KEY not in str(refused.value)
        assert refused.value.__cause__ is None

    def test_a_write_refuses_a_stored_value_that_does_not_parse(self):
        settings.set_value(cfg.data_root, "server_presets", f"not json {KEY}")
        with pytest.raises(ValueError, match="do not parse") as refused:
            presets.add_preset("alpha", URL, roles=["chat"])
        assert KEY not in str(refused.value)


class TestNoGeneralPath:
    """Only this service writes or reads the stored list; the settings surfaces refuse it."""

    FIELD = "server_presets"
    REFUSAL = "Unknown or read-only setting: server_presets"

    @pytest.fixture(autouse=True)
    def _alpha(self):
        presets.add_preset("alpha", URL, roles=["chat"], api_key=KEY)
        self.before = (cfg.data_root / "config.toml").read_bytes()

    def test_the_field_stays_writable_so_the_overlay_loads_it(self):
        assert self.FIELD in WRITABLE_CONFIG_FIELDS
        assert self.FIELD in SERVICE_ONLY_FIELDS

    def test_a_settings_update_is_refused(self):
        with pytest.raises(ValueError, match=self.REFUSAL):
            app_settings.apply_settings_update({self.FIELD: "[]"})
        assert (cfg.data_root / "config.toml").read_bytes() == self.before

    def test_a_settings_reset_is_refused(self):
        with pytest.raises(ValueError, match=self.REFUSAL):
            app_settings.reset_settings([self.FIELD])
        assert (cfg.data_root / "config.toml").read_bytes() == self.before

    def test_a_settings_read_is_refused(self):
        with pytest.raises(KeyError, match=self.REFUSAL):
            app_settings.get_setting(self.FIELD)

    def test_the_settings_list_leaves_it_out_and_holds_no_key(self):
        listed = app_settings.list_settings()
        assert len(listed) > 20
        assert self.FIELD not in {info.key for info in listed}
        assert KEY not in repr(listed)

    async def test_the_http_config_read_leaves_it_out_and_holds_no_key(self):
        assert cfg.server_presets is not None
        assert KEY in cfg.server_presets
        for payload in (await get_config(), await get_config_defaults()):
            dumped = payload.model_dump()
            assert len(dumped) > 20
            assert self.FIELD not in dumped
            assert KEY not in repr(dumped)
