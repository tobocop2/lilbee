"""Tests for the llama-swap config generator."""

from __future__ import annotations

import json

import pytest

from lilbee.providers.fleet import swap_config as swap_config_mod
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.swap_config import build_swap_config
from lilbee.providers.roles import WorkerRole


def _mid(role: WorkerRole, replica: int = 0) -> str:
    """The llama-swap model id for a role's replica (matches InstanceLaunch.model_id)."""
    return f"{role.value}-{replica}"


def _launch(
    role: WorkerRole,
    argv: list[str],
    env: dict[str, str] | None = None,
    replica: int = 0,
    swap_class: str = "",
) -> InstanceLaunch:
    return InstanceLaunch(
        role=role,
        replica=replica,
        argv=argv,
        env_overrides=env or {},
        model=f"{role.value}-model",
        swap_class=swap_class,
    )


_BASE_PORT = 5900


def _config(launches: list[InstanceLaunch]) -> dict:
    ports = {launch.model_id: _BASE_PORT + i for i, launch in enumerate(launches)}
    return json.loads(build_swap_config(launches, ports, engine_logs={}))


def test_instance_launch_rerank_mode_defaults_none() -> None:
    assert _launch(WorkerRole.RERANK, ["llama-server"]).rerank_mode is None


class TestBuildSwapConfig:
    def test_emits_valid_json_with_top_level_keys(self) -> None:
        cfg = _config([_launch(WorkerRole.CHAT, ["/bin/llama-server", "--model", "/m/c.gguf"])])
        assert "startPort" not in cfg  # members carry explicit ports, never a fixed range
        assert cfg["healthCheckTimeout"] >= 1
        assert "logLevel" in cfg

    def test_one_model_per_launch_keyed_by_model_id(self) -> None:
        cfg = _config(
            [
                _launch(WorkerRole.CHAT, ["/bin/llama-server", "--jinja"]),
                _launch(WorkerRole.EMBED, ["/bin/llama-server", "--embeddings"]),
            ]
        )
        assert set(cfg["models"]) == {_mid(WorkerRole.CHAT), _mid(WorkerRole.EMBED)}

    def test_replicas_get_distinct_model_entries_all_co_resident(self) -> None:
        # Each embed replica is its own llama-swap model id, all held in the group.
        cfg = _config(
            [
                _launch(WorkerRole.CHAT, ["/bin/llama-server"]),
                _launch(WorkerRole.EMBED, ["/bin/llama-server", "--embeddings"], replica=0),
                _launch(WorkerRole.EMBED, ["/bin/llama-server", "--embeddings"], replica=1),
            ]
        )
        assert set(cfg["models"]) == {
            _mid(WorkerRole.CHAT),
            _mid(WorkerRole.EMBED, 0),
            _mid(WorkerRole.EMBED, 1),
        }
        (group,) = cfg["groups"].values()
        assert _mid(WorkerRole.EMBED, 0) in group["members"]
        assert _mid(WorkerRole.EMBED, 1) in group["members"]

    def test_group_holds_all_roles_co_resident(self) -> None:
        cfg = _config(
            [
                _launch(WorkerRole.CHAT, ["/bin/llama-server"]),
                _launch(WorkerRole.EMBED, ["/bin/llama-server"]),
                _launch(WorkerRole.RERANK, ["/bin/llama-server"]),
            ]
        )
        (group,) = cfg["groups"].values()
        assert group["swap"] is False  # never evict a member to load another
        assert group["persistent"] is True
        assert set(group["members"]) == {
            _mid(WorkerRole.CHAT),
            _mid(WorkerRole.EMBED),
            _mid(WorkerRole.RERANK),
        }

    def test_launches_without_a_class_each_evict_the_others(self) -> None:
        # Exclusive groups make llama-swap unload chat to load vision, and back.
        cfg = json.loads(
            build_swap_config(
                [
                    _launch(WorkerRole.CHAT, ["/bin/llama-server"]),
                    _launch(WorkerRole.VISION, ["/bin/llama-server"]),
                ],
                {_mid(WorkerRole.CHAT): 1, _mid(WorkerRole.VISION): 2},
                swap=True,
                engine_logs={},
            )
        )
        assert {name: group["members"] for name, group in cfg["groups"].items()} == {
            f"lilbee-{_mid(WorkerRole.CHAT)}": [_mid(WorkerRole.CHAT)],
            f"lilbee-{_mid(WorkerRole.VISION)}": [_mid(WorkerRole.VISION)],
        }
        for group in cfg["groups"].values():
            assert (group["swap"], group["exclusive"], group["persistent"]) == (False, True, False)

    def test_launches_of_one_class_stay_in_one_group(self) -> None:
        cfg = json.loads(
            build_swap_config(
                [
                    _launch(WorkerRole.CHAT, ["/bin/llama-server"], swap_class="0"),
                    _launch(WorkerRole.RERANK, ["/bin/llama-server"], swap_class="0"),
                    _launch(WorkerRole.VISION, ["/bin/llama-server"], swap_class="1"),
                ],
                {_mid(WorkerRole.CHAT): 1, _mid(WorkerRole.RERANK): 2, _mid(WorkerRole.VISION): 3},
                swap=True,
                engine_logs={},
            )
        )
        assert {name: group["members"] for name, group in cfg["groups"].items()} == {
            "lilbee-0": [_mid(WorkerRole.CHAT), _mid(WorkerRole.RERANK)],
            "lilbee-1": [_mid(WorkerRole.VISION)],
        }
        assert all(group["exclusive"] is True for group in cfg["groups"].values())

    def test_a_class_does_not_split_the_group_of_a_process_without_swap(self) -> None:
        launches = [
            _launch(WorkerRole.CHAT, ["/bin/llama-server"], swap_class="0"),
            _launch(WorkerRole.VISION, ["/bin/llama-server"], swap_class="1"),
        ]
        ports = {_mid(WorkerRole.CHAT): 1, _mid(WorkerRole.VISION): 2}
        cfg = json.loads(build_swap_config(launches, ports, engine_logs={}))
        (group,) = cfg["groups"].values()
        assert set(group["members"]) == {_mid(WorkerRole.CHAT), _mid(WorkerRole.VISION)}

    def test_command_carries_explicit_port_and_role_argv(self) -> None:
        cfg = _config([_launch(WorkerRole.CHAT, ["/bin/llama-server", "--jinja"])])
        cmd = cfg["models"][_mid(WorkerRole.CHAT)]["cmd"]
        assert "--jinja" in cmd
        assert cmd.endswith(f"--port {_BASE_PORT}")
        assert cfg["models"][_mid(WorkerRole.CHAT)]["proxy"] == f"http://127.0.0.1:{_BASE_PORT}"

    def test_each_member_gets_its_own_port(self) -> None:
        cfg = _config(
            [
                _launch(WorkerRole.CHAT, ["/bin/llama-server"]),
                _launch(WorkerRole.EMBED, ["/bin/llama-server", "--embeddings"]),
            ]
        )
        chat = cfg["models"][_mid(WorkerRole.CHAT)]
        embed = cfg["models"][_mid(WorkerRole.EMBED)]
        assert chat["cmd"].endswith(f"--port {_BASE_PORT}")
        assert embed["cmd"].endswith(f"--port {_BASE_PORT + 1}")
        assert embed["proxy"] == f"http://127.0.0.1:{_BASE_PORT + 1}"

    def test_embed_command_keeps_embeddings_flag(self) -> None:
        cfg = _config([_launch(WorkerRole.EMBED, ["/bin/llama-server", "--embeddings"])])
        assert "--embeddings" in cfg["models"][_mid(WorkerRole.EMBED)]["cmd"]

    def test_spaced_model_path_is_quoted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(swap_config_mod.sys, "platform", "linux")
        argv = ["/bin/llama-server", "--model", "/Application Support/lilbee/m.gguf"]
        cfg = _config([_launch(WorkerRole.CHAT, argv)])
        cmd = cfg["models"][_mid(WorkerRole.CHAT)]["cmd"]
        assert "'/Application Support/lilbee/m.gguf'" in cmd  # shell-quoted, survives the space

    def test_windows_uses_ms_quoting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # llama-swap splits cmd with MS rules on Windows; POSIX single quotes
        # would stay literal in the paths and the server spawn fails.
        monkeypatch.setattr(swap_config_mod.sys, "platform", "win32")
        argv = ["C:\\llama\\llama-server.exe", "--model", "C:\\Program Files\\lilbee\\m.gguf"]
        cmd = _config([_launch(WorkerRole.CHAT, argv)])["models"][_mid(WorkerRole.CHAT)]["cmd"]
        assert "'" not in cmd
        assert '"C:\\Program Files\\lilbee\\m.gguf"' in cmd
        assert cmd.startswith("C:\\llama\\llama-server.exe")

    def test_env_overrides_become_env_list_when_present(self) -> None:
        cfg = _config(
            [_launch(WorkerRole.CHAT, ["/bin/llama-server"], env={"CUDA_VISIBLE_DEVICES": "0"})]
        )
        assert cfg["models"][_mid(WorkerRole.CHAT)]["env"] == ["CUDA_VISIBLE_DEVICES=0"]

    def test_env_key_absent_when_no_overrides(self) -> None:
        cfg = _config([_launch(WorkerRole.CHAT, ["/bin/llama-server"])])
        assert "env" not in cfg["models"][_mid(WorkerRole.CHAT)]

    def test_a_member_with_its_own_log_keeps_its_output_out_of_the_group_log(
        self, tmp_path
    ) -> None:
        launch = _launch(WorkerRole.CHAT, ["/bin/llama-server"])
        log_path = tmp_path / "engine-chat-0.log"
        ports = {launch.model_id: _BASE_PORT}
        cfg = json.loads(
            build_swap_config([launch], ports, engine_logs={launch.model_id: log_path})
        )
        assert cfg["logToStdout"] == "proxy"
        assert f"LLAMA_LOG_FILE={log_path}" in cfg["models"][launch.model_id]["env"]

    def test_a_member_without_its_own_log_has_its_output_in_the_group_log(self) -> None:
        cfg = _config([_launch(WorkerRole.EMBED, ["/bin/text-embeddings-router"])])
        assert cfg["logToStdout"] == "proxy,upstream"
        assert "env" not in cfg["models"][_mid(WorkerRole.EMBED)]

    def test_one_member_without_its_own_log_copies_the_output_of_the_group(self, tmp_path) -> None:
        chat = _launch(WorkerRole.CHAT, ["/bin/llama-server"])
        embed = _launch(WorkerRole.EMBED, ["/bin/text-embeddings-router"])
        ports = {chat.model_id: _BASE_PORT, embed.model_id: _BASE_PORT + 1}
        engine_logs = {chat.model_id: tmp_path / "engine-chat-0.log"}
        cfg = json.loads(build_swap_config([chat, embed], ports, engine_logs=engine_logs))
        assert cfg["logToStdout"] == "proxy,upstream"

    def test_a_router_that_reads_no_stream_list_gets_the_proxy_stream_alone(self) -> None:
        launch = _launch(WorkerRole.EMBED, ["/bin/text-embeddings-router"])
        rendered = build_swap_config(
            [launch], {launch.model_id: _BASE_PORT}, engine_logs={}, router_reads_streams=False
        )
        assert json.loads(rendered)["logToStdout"] == "proxy"

    def test_the_engine_logs_are_required(self) -> None:
        launch = _launch(WorkerRole.CHAT, ["/bin/llama-server"])
        with pytest.raises(TypeError, match="engine_logs"):
            build_swap_config([launch], {launch.model_id: _BASE_PORT})  # type: ignore[call-arg]

    def test_member_never_times_out(self) -> None:
        cfg = _config([_launch(WorkerRole.CHAT, ["/bin/llama-server"])])
        assert cfg["models"][_mid(WorkerRole.CHAT)]["ttl"] == 0

    def test_ttl_seconds_flows_to_every_member(self) -> None:
        """llama-swap's ttl is in seconds; a warm fleet unloads idle weights after it."""
        launches = [_launch(WorkerRole.CHAT, ["/bin/llama-server"])]
        ports = {launch.model_id: _BASE_PORT + i for i, launch in enumerate(launches)}
        cfg = json.loads(build_swap_config(launches, ports, ttl_seconds=900, engine_logs={}))
        assert cfg["models"][_mid(WorkerRole.CHAT)]["ttl"] == 900


class TestHealthCheckTimeoutScaling:
    @pytest.mark.parametrize(
        ("role", "floor_s"),
        [
            (WorkerRole.CHAT, 600),
            (WorkerRole.VISION, 600),
            (WorkerRole.RERANK, 600),
            (WorkerRole.EMBED, 120),
        ],
    )
    def test_small_model_gets_its_role_floor(self, role: WorkerRole, floor_s: int) -> None:
        # An embedder's weights load in seconds, so llama-swap gives up on a wedged
        # embed restart after two minutes; the generating roles keep ten.
        launch = _launch(role, ["/bin/llama-server"])
        launch.weights_bytes = 4 * 1024**3
        cfg = _config([launch])
        assert cfg["healthCheckTimeout"] == floor_s

    def test_giant_model_scales_the_timeout_past_the_floor(self) -> None:
        # 300 GB at the conservative 150 MB/s disk rate needs ~2048s, not 600s;
        # llama-swap would otherwise kill the server mid-load.
        launch = _launch(WorkerRole.CHAT, ["/bin/llama-server"])
        launch.weights_bytes = 300 * 1024**3
        cfg = _config([launch])
        assert cfg["healthCheckTimeout"] == 2048

    def test_giant_embedder_scales_past_its_floor(self) -> None:
        # 30 GB streams in ~204s at the conservative rate, past the embed floor.
        launch = _launch(WorkerRole.EMBED, ["/bin/llama-server"])
        launch.weights_bytes = 30 * 1024**3
        cfg = _config([launch])
        assert cfg["healthCheckTimeout"] == 204

    def test_heaviest_member_sets_the_proxy_global_timeout(self) -> None:
        small = _launch(WorkerRole.EMBED, ["/bin/llama-server"])
        small.weights_bytes = 1 * 1024**3
        giant = _launch(WorkerRole.CHAT, ["/bin/llama-server"])
        giant.weights_bytes = 300 * 1024**3
        cfg = _config([small, giant])
        assert cfg["healthCheckTimeout"] == 2048

    def test_co_tenant_embedder_takes_the_slower_role_floor(self) -> None:
        # The timeout is proxy-global: an embedder sharing a group with a small
        # reranker must not cut the reranker's load short at the embed floor.
        embed = _launch(WorkerRole.EMBED, ["/bin/llama-server"])
        embed.weights_bytes = 1 * 1024**3
        rerank = _launch(WorkerRole.RERANK, ["/bin/llama-server"])
        rerank.weights_bytes = 1 * 1024**3
        cfg = _config([embed, rerank])
        assert cfg["healthCheckTimeout"] == 600

    def test_no_members_takes_the_longest_floor(self) -> None:
        assert _config([])["healthCheckTimeout"] == 600

    def test_cold_load_timeout_floors_small_weights_per_role(self) -> None:
        # The shared per-member helper; the provider's client timeout derives from it.
        from lilbee.providers.fleet.swap_config import cold_load_timeout_s

        assert cold_load_timeout_s(0, WorkerRole.CHAT, 0) == 600
        assert cold_load_timeout_s(4 * 1024**3, WorkerRole.CHAT, 0) == 600
        assert cold_load_timeout_s(0, WorkerRole.EMBED, 0) == 120
        assert cold_load_timeout_s(4 * 1024**3, WorkerRole.EMBED, 0) == 120

    def test_cold_load_timeout_scales_giant_weights(self) -> None:
        from lilbee.providers.fleet.swap_config import cold_load_timeout_s

        weights = 300 * 1024**3
        assert cold_load_timeout_s(weights, WorkerRole.CHAT, 0) == 2048
        assert cold_load_timeout_s(weights, WorkerRole.EMBED, 0) == 2048

    @pytest.mark.parametrize(("weights_gib", "expected"), [(20, 1638), (60, 3600)])
    def test_a_slow_measured_volume_stretches_the_cold_load(
        self, weights_gib: int, expected: int
    ) -> None:
        # A 25 MiB/s volume, scaled at half its measured rate: a 20 GiB load that
        # needs about 819 s at that rate is not given up on at the 600 s floor.
        from lilbee.providers.fleet.swap_config import cold_load_timeout_s

        rate = 25 * 1024**2
        assert cold_load_timeout_s(weights_gib * 1024**3, WorkerRole.CHAT, rate) == expected

    def test_a_fast_measured_volume_keeps_the_conservative_rate(self) -> None:
        # A sample that hit the page cache reads far above any disk; it must not
        # shorten the wait below the conservative rate.
        from lilbee.providers.fleet.swap_config import cold_load_timeout_s

        weights = 300 * 1024**3
        assert cold_load_timeout_s(weights, WorkerRole.CHAT, 8 * 1024**3) == 2048

    @pytest.mark.parametrize(
        ("rate", "expected"),
        [
            pytest.param(8738, 2048, id="stalled-sample"),
            pytest.param(1, 2048, id="rate-of-one"),
            pytest.param(0, 600, id="unmeasured"),
        ],
    )
    def test_a_slow_reading_is_floored_at_the_slowest_believable_rate(
        self, rate: int, expected: int
    ) -> None:
        # One 1 MiB chunk that stalls 120 s inside the sample reads as 8738 B/s.
        # The wait uses 10 MiB/s at the slowest: 20 GiB gives 2048 s, not 57 days.
        from lilbee.providers.fleet.swap_config import cold_load_timeout_s

        assert cold_load_timeout_s(20 * 1024**3, WorkerRole.CHAT, rate) == expected

    @pytest.mark.parametrize("rate", [1, 8738, 25 * 1024**2])
    def test_a_stretched_wait_stops_at_the_cap(self, rate: int) -> None:
        from lilbee.providers.fleet.swap_config import _MEASURED_WAIT_CAP_S, cold_load_timeout_s

        assert cold_load_timeout_s(60 * 1024**3, WorkerRole.CHAT, rate) == _MEASURED_WAIT_CAP_S
        assert cold_load_timeout_s(60 * 1024**3, WorkerRole.EMBED, rate) == _MEASURED_WAIT_CAP_S

    def test_a_model_whose_base_wait_passes_the_cap_keeps_its_base_wait(self) -> None:
        # The cap bounds what a measurement adds; it never cuts the base wait.
        from lilbee.providers.fleet.swap_config import cold_load_timeout_s

        weights = 1024 * 1024**3
        assert cold_load_timeout_s(weights, WorkerRole.CHAT, 0) == 6990
        assert cold_load_timeout_s(weights, WorkerRole.CHAT, 1) == 6990

    def test_the_slowest_measured_member_sets_the_proxy_global_timeout(self) -> None:
        member = _launch(WorkerRole.CHAT, ["/bin/llama-server"])
        member.weights_bytes = 20 * 1024**3
        member.read_bytes_per_s = 25 * 1024**2
        assert _config([member])["healthCheckTimeout"] == 1638

    def test_a_stalled_member_sets_the_proxy_global_timeout_at_the_cap(self) -> None:
        member = _launch(WorkerRole.CHAT, ["/bin/llama-server"])
        member.weights_bytes = 60 * 1024**3
        member.read_bytes_per_s = 1
        assert _config([member])["healthCheckTimeout"] == 3600


class TestWarmTtlSeconds:
    """llama-swap's idle timer: 0 wherever the reaper owns the unload, else the idle minutes."""

    @pytest.fixture(autouse=True)
    def _idle_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.providers.fleet import provider as prov_mod

        monkeypatch.setattr(cfg, "engine_idle_ttl_minutes", 15)
        monkeypatch.setattr(prov_mod, "kernel_arbitrates_locks", lambda _d: True)
        monkeypatch.setattr(prov_mod, "ensure_engine_reaper", lambda _d: True)

    def test_a_kept_warm_build_without_a_reaper_keeps_the_idle_window(self, monkeypatch, tmp_path):
        # Nothing would unload weights held by ttl 0 once lilbee exits.
        from lilbee.core.config import cfg
        from lilbee.providers.fleet import provider as prov_mod
        from lilbee.providers.fleet.swap_manager import EngineLifetime
        from lilbee.runtime.engine_lock import request_keep_warm

        monkeypatch.setattr(prov_mod, "ensure_engine_reaper", lambda _d: False)
        monkeypatch.setattr(cfg, "keep_engine_warm", True)
        assert prov_mod._warm_ttl_seconds(EngineLifetime.DETACHED, tmp_path) == 900
        monkeypatch.setattr(cfg, "keep_engine_warm", False)
        request_keep_warm(tmp_path, tmp_path / "peer-install", 5)
        assert prov_mod._warm_ttl_seconds(EngineLifetime.USERS, tmp_path) == 900

    def test_a_warm_off_build_starts_no_reaper_for_the_ttl(self, monkeypatch, tmp_path):
        from lilbee.core.config import cfg
        from lilbee.providers.fleet import provider as prov_mod
        from lilbee.providers.fleet.swap_manager import EngineLifetime

        asked: list[object] = []
        monkeypatch.setattr(prov_mod, "ensure_engine_reaper", lambda d: asked.append(d) or True)
        monkeypatch.setattr(cfg, "keep_engine_warm", False)
        assert prov_mod._warm_ttl_seconds(EngineLifetime.USERS, tmp_path) == 900
        assert asked == []

    def test_keep_engine_warm_holds_weights_while_users_remain(self, monkeypatch, tmp_path):
        # The reaper unloads a keep-warm engine after the last user, so llama-swap never does.
        from lilbee.core.config import cfg
        from lilbee.providers.fleet.provider import _warm_ttl_seconds
        from lilbee.providers.fleet.swap_manager import EngineLifetime

        monkeypatch.setattr(cfg, "keep_engine_warm", True)
        assert _warm_ttl_seconds(EngineLifetime.DETACHED, tmp_path) == 0

    def test_a_peers_keep_warm_marker_holds_weights_for_a_warm_off_builder(
        self, monkeypatch, tmp_path
    ):
        from lilbee.core.config import cfg
        from lilbee.providers.fleet.provider import _warm_ttl_seconds
        from lilbee.providers.fleet.swap_manager import EngineLifetime
        from lilbee.runtime.engine_lock import request_keep_warm

        monkeypatch.setattr(cfg, "keep_engine_warm", False)
        assert _warm_ttl_seconds(EngineLifetime.USERS, tmp_path) == 900
        request_keep_warm(tmp_path, tmp_path / "peer-install", 5)
        assert _warm_ttl_seconds(EngineLifetime.USERS, tmp_path) == 0

    def test_keep_engine_warm_without_kernel_locks_keeps_the_idle_window(
        self, monkeypatch, tmp_path
    ):
        # No user can be counted, so no reaper can unload: llama-swap's timer stays armed.
        from lilbee.core.config import cfg
        from lilbee.providers.fleet import provider as prov_mod
        from lilbee.providers.fleet.swap_manager import EngineLifetime

        monkeypatch.setattr(cfg, "keep_engine_warm", True)
        monkeypatch.setattr(prov_mod, "kernel_arbitrates_locks", lambda _d: False)
        assert prov_mod._warm_ttl_seconds(EngineLifetime.DETACHED, tmp_path) == 900

    def test_interactive_session_pins_weights_resident(self, monkeypatch, tmp_path) -> None:
        # A provider serving an interactive session (the TUI) holds its bound
        # engine warm for its whole lifetime, regardless of the idle window.
        from lilbee.core.config import cfg
        from lilbee.providers.fleet.provider import _warm_ttl_seconds
        from lilbee.providers.fleet.swap_manager import EngineLifetime

        monkeypatch.setattr(cfg, "keep_engine_warm", False)
        hold = {"hold_warm_for_session": True}
        assert _warm_ttl_seconds(EngineLifetime.BUILDER, tmp_path, **hold) == 0
        # A provider not serving one still unloads on the idle window.
        assert _warm_ttl_seconds(EngineLifetime.BUILDER, tmp_path) == 900
        # An engine shared by its users outlives the session, so the hold is off.
        assert _warm_ttl_seconds(EngineLifetime.USERS, tmp_path, **hold) == 900

    def test_on_demand_unloads_after_idle_ttl(self, monkeypatch, tmp_path) -> None:
        # No interactive session and no keep_engine_warm: the fleet releases its
        # weights after the idle window.
        from lilbee.core.config import cfg
        from lilbee.providers.fleet.provider import _warm_ttl_seconds
        from lilbee.providers.fleet.swap_manager import EngineLifetime

        monkeypatch.setattr(cfg, "keep_engine_warm", False)
        assert _warm_ttl_seconds(EngineLifetime.BUILDER, tmp_path) == 900

    def test_on_demand_with_zero_ttl_keeps_weights_until_teardown(
        self, monkeypatch, tmp_path
    ) -> None:
        # 0 disables the idle unload even for an on-demand fleet.
        from lilbee.core.config import cfg
        from lilbee.providers.fleet.provider import _warm_ttl_seconds
        from lilbee.providers.fleet.swap_manager import EngineLifetime

        monkeypatch.setattr(cfg, "keep_engine_warm", False)
        monkeypatch.setattr(cfg, "engine_idle_ttl_minutes", 0)
        assert _warm_ttl_seconds(EngineLifetime.BUILDER, tmp_path) == 0


@pytest.mark.parametrize(
    ("role", "slots"),
    [
        (WorkerRole.CHAT, 4),
        (WorkerRole.CHAT, 40),
        (WorkerRole.VISION, 8),
        (WorkerRole.EMBED, 1),
        (WorkerRole.RERANK, 8),
    ],
    ids=["chat", "chat-many-slots", "vision", "embed", "rerank"],
)
def test_every_model_carries_the_same_fixed_concurrency_limit(role: WorkerRole, slots: int) -> None:
    # llama-swap answers 429 past the limit (default 10) and never queues. The limit is
    # one number for every model: it reads no slot count, pool size or other process.
    launch = InstanceLaunch(
        role=role, replica=0, argv=["/bin/llama-server"], env_overrides={}, model="m", slots=slots
    )
    entry = _config([launch])["models"][launch.model_id]
    assert entry["concurrencyLimit"] == 1024
    assert swap_config_mod.CONCURRENCY_LIMIT == 1024


def test_the_config_reads_no_ingest_pool_size() -> None:
    assert not hasattr(swap_config_mod, "max_workers")
