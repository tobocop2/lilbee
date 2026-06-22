"""Model bar: a horizontal band of the four role pickers plus the Search/Chat toggle."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, NamedTuple

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.model_picker import PickerScope
    from lilbee.modelhub.registry import ModelRegistry

from textual import events, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Static

from lilbee.app.services import get_services, reset_services
from lilbee.catalog import clean_display_name, display_label_for_ref, extract_quant
from lilbee.catalog.types import ModelTask
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import apply_setting
from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.screens.settings_widgets import model_field_to_picker_scope
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.model_pick import apply_model_pick, config_key_for_scope
from lilbee.core.config import cfg
from lilbee.core.config.enums import ChatMode
from lilbee.providers.model_ref import format_remote_ref, parse_model_ref
from lilbee.providers.sdk_backend import PROVIDER_KEYS
from lilbee.retrieval.embedder import is_model_available

log = logging.getLogger(__name__)

_MMPROJ_MARKER = "mmproj"

# Routing-name -> display-label map derived from PROVIDER_KEYS. Any new
# entry added there lights up the warning without further changes here.
_CLOUD_PROVIDER_LABELS: dict[str, str] = {name: label for name, _, _, label in PROVIDER_KEYS}


def _cloud_provider_label(chat_model: str) -> str | None:
    """Return the provider display label for cloud-routed models, else None."""
    if not chat_model:
        return None
    ref = parse_model_ref(chat_model)
    if not ref.is_api:
        return None
    return _CLOUD_PROVIDER_LABELS.get(ref.provider)


class ModelOption(NamedTuple):
    """A selectable model with display label and config ref."""

    label: str  # human-readable name for the dropdown
    ref: str  # canonical ref persisted to config


def _is_mmproj(name: str) -> bool:
    """Return True if a model name refers to an mmproj projection file."""
    return _MMPROJ_MARKER in name.lower()


def classify_installed_models_full() -> dict[ModelTask, list[ModelOption]]:
    """Classify installed models into per-task lists, dropping mmproj entries."""
    buckets: dict[ModelTask, list[ModelOption]] = {task: [] for task in ModelTask}
    seen: set[str] = set()

    _collect_native_models(buckets, seen)
    _collect_remote_models(buckets, seen)
    _collect_api_models(buckets, seen)

    return {task: sorted(opts, key=lambda o: o.ref) for task, opts in buckets.items()}


def _lookup_bucket(
    buckets: dict[ModelTask, list[ModelOption]], task: str, ref: str
) -> list[ModelOption] | None:
    """Return the bucket for *task*, or None if it is not a known ModelTask."""
    try:
        key = ModelTask(task)
    except ValueError:
        log.debug("dropping %r with unknown task %r", ref, task)
        return None
    return buckets.get(key)


def _native_label(hf_repo: str, gguf_filename: str, repo_count: int) -> str:
    """Build the picker label, appending the quant suffix only on collision."""
    base = clean_display_name(hf_repo)
    if repo_count <= 1:
        return base
    quant = extract_quant(gguf_filename)
    return f"{base} ({quant})" if quant else base


def _has_vision_sidecar(registry: ModelRegistry, ref: str) -> bool:
    """Return True if *ref* resolves to a model with an adjacent ``*mmproj*.gguf`` file.

    Models like ``google/gemma-3-12b-it`` carry their vision capability in
    a sibling ``mmproj`` GGUF; without checking the file system, the
    ref's name alone gives no signal that the model is multimodal, so the
    vision picker would silently miss it.
    """
    try:
        path = registry.resolve(ref)
    except (KeyError, ValueError):
        return False
    return any(path.parent.glob("*mmproj*.gguf"))


def _collect_native_models(buckets: dict[ModelTask, list[ModelOption]], seen: set[str]) -> None:
    """Add native registry models to buckets."""
    try:
        from lilbee.modelhub.registry import ModelRegistry

        registry = ModelRegistry(cfg.models_dir)
        manifests = registry.list_installed()
        repo_counts: dict[str, int] = {}
        for m in manifests:
            repo_counts[m.hf_repo] = repo_counts.get(m.hf_repo, 0) + 1

        from lilbee.catalog.query import reclassify_by_name

        for manifest in manifests:
            ref = manifest.ref
            if _is_mmproj(manifest.gguf_filename) or ref in seen:
                continue
            task = reclassify_by_name(ref, manifest.task)
            label = _native_label(
                manifest.hf_repo, manifest.gguf_filename, repo_counts[manifest.hf_repo]
            )
            primary_bucket = _lookup_bucket(buckets, task, ref)
            if primary_bucket is None:
                continue
            seen.add(ref)
            primary_bucket.append(ModelOption(label=label, ref=ref))
            # If the model has an mmproj sidecar it is also vision-capable.
            # Surface it under the vision picker too without dropping its
            # primary classification, so a chat model with vision (e.g.
            # gemma-3 with mmproj) shows up in both pickers.
            if task != ModelTask.VISION and _has_vision_sidecar(registry, ref):
                buckets[ModelTask.VISION].append(ModelOption(label=label, ref=ref))
    except Exception:
        log.debug("Could not read native model registry", exc_info=True)


def _collect_remote_models(buckets: dict[ModelTask, list[ModelOption]], seen: set[str]) -> None:
    """Add remote (Ollama / OpenAI-compatible) models, prefixed for routing.

    Skipped when the litellm extra is not installed -- surfacing a model
    the SDK cannot route is a guaranteed runtime error.
    """
    from lilbee.providers.litellm_sdk import litellm_available

    if not litellm_available():
        return
    try:
        from lilbee.modelhub.model_manager import classify_all_remote_models

        for model in classify_all_remote_models():
            # Skip backend rows with a blank model name so the picker
            # doesn't render an empty " (Ollama)" row.
            if not model.name.strip():
                continue
            ref = format_remote_ref(model.name, model.provider)
            if ref in seen or _is_mmproj(model.name):
                continue
            bucket = _lookup_bucket(buckets, model.task, ref)
            if bucket is None:
                continue
            seen.add(ref)
            label = f"{model.name} ({model.provider})"
            bucket.append(ModelOption(label=label, ref=ref))
    except Exception:
        log.debug("Could not classify remote models", exc_info=True)


def _collect_api_models(buckets: dict[ModelTask, list[ModelOption]], seen: set[str]) -> None:
    """Add frontier API chat models. Skipped without litellm (cannot route)."""
    from lilbee.providers.litellm_sdk import litellm_available

    if not litellm_available():
        return
    try:
        from lilbee.modelhub.model_manager import discover_api_models

        # API discovery returns only chat-capable refs; revisit if providers
        # expose embedding/vision/rerank.
        for display_name, models in discover_api_models().items():
            for model in models:
                qualified = format_remote_ref(model.name, model.provider)
                if qualified in seen:
                    continue
                seen.add(qualified)
                label = f"{model.name} ({display_name})"
                buckets[ModelTask.CHAT].append(ModelOption(label=label, ref=qualified))
    except Exception:
        log.debug("Could not discover API models", exc_info=True)


_CHAT_MODE_TOGGLE_ID = "chat-mode-toggle"
_CHAT_MODE_SEARCH_PILL_ID = "chat-mode-search"
_CHAT_MODE_CHAT_PILL_ID = "chat-mode-chat"
_CHAT_MODE_PILL_CLASS = "chat-mode-pill"
_CHAT_MODE_DISABLED_CLASS = "-disabled"
_CHAT_MODE_ACTIVE_CLASS = "-active"


_SCOPE_TO_TOOLTIP: dict[str, str] = {
    "chat": msg.MODEL_PICKER_CHAT_TOOLTIP,
    "embed": msg.MODEL_PICKER_EMBED_TOOLTIP,
    "vision": msg.MODEL_PICKER_VISION_TOOLTIP,
    "rerank": msg.MODEL_PICKER_RERANK_TOOLTIP,
}

_CSS_FILE = Path(__file__).parent / "model_bar.tcss"

_CLOUD_WARNING_ID = "model-bar-cloud-warning"

_SCOPE_TO_LABEL: dict[str, str] = {
    "chat": msg.MODEL_BAR_CHAT_LABEL,
    "embed": msg.MODEL_BAR_EMBED_LABEL,
    "vision": msg.MODEL_BAR_VISION_LABEL,
    "rerank": msg.MODEL_BAR_RERANK_LABEL,
}

# Per-role pill colors (background, foreground) when the role is active. Chat and
# Embed mirror the original bar; Vision and Rerank get their own accent hues.
_SCOPE_PILL_COLORS: dict[str, tuple[str, str]] = {
    "chat": ("$primary", "$text"),
    "embed": ("$secondary", "$text"),
    "vision": ("#bc8cff", "$text"),
    "rerank": ("#f0883e", "$text"),
}

# Muted pill for an optional role that is currently off.
_OFF_PILL_COLORS: tuple[str, str] = ("$surface-lighten-2", "$text-muted")


class ModelPickerButton(Static, can_focus=True):
    """Pill button that opens a ModelPickerModal scoped to one of the four roles."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "open_picker", "Pick model", show=False),
        Binding("space", "open_picker", "Pick model", show=False),
    ]

    def __init__(self, *, scope: PickerScope, button_id: str) -> None:
        super().__init__(id=button_id)
        self._scope: PickerScope = scope
        self._key: str = config_key_for_scope(scope)
        self._options: list[ModelOption] = []
        self.tooltip = _SCOPE_TO_TOOLTIP[scope]

    def on_mount(self) -> None:
        self._refresh()

    def set_options(self, options: list[ModelOption]) -> None:
        """Update the options pool. Repaints the label from cfg."""
        self._options = options
        if self.is_mounted:
            self._refresh()

    def _refresh(self) -> None:
        # Only optional roles (vision/rerank) can be empty; chat/embed are
        # non-nullable, so an empty ref means the role is off, not a model
        # called "(none)".
        ref = getattr(cfg, self._key)
        label = (display_label_for_ref(ref) or ref) if ref else msg.MODEL_BAR_DISABLED
        self.update(label)

    def repaint(self) -> None:
        """Public entry for a parent container to repaint the label from cfg."""
        self._refresh()

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.open_picker()

    def action_open_picker(self) -> None:
        self.open_picker()

    def _is_nullable(self) -> bool:
        from lilbee.app.settings_map import SETTINGS_MAP

        defn = SETTINGS_MAP.get(self._key)
        return defn is not None and defn.nullable

    def open_picker(self) -> None:
        # Lazy import: model_picker imports ModelOption from this module.
        from lilbee.cli.tui.screens.model_picker import ModelPickerModal

        options = list(self._options)
        # Optional role that's on: offer an explicit "turn off" action in the
        # modal (ref "" disables it) so disabling isn't only a pill click.
        if self._is_nullable() and getattr(cfg, self._key):
            options.append(ModelOption(label=msg.MODEL_PICKER_TURN_OFF, ref=""))
        modal = ModelPickerModal(scope=self._scope, options=options)
        self.app.push_screen(modal, self._on_picker_dismissed)

    def _on_picker_dismissed(self, ref: str | None) -> None:
        if ref is not None and ref == getattr(cfg, self._key):
            return
        # Chat swaps reset services in _commit_after_change -> apply_model_change,
        # so the helper must not also reload the chat worker (double teardown).
        apply_model_pick(
            self,
            key=self._key,
            ref=ref,
            on_done=self._commit_after_change,
            reload_worker=self._scope != "chat",
        )

    def _commit_after_change(self) -> None:
        """Repaint the label, then run the chat-screen side effect for chat swaps.

        ``apply_model_pick`` already persisted the ref and (for non-chat
        scopes) reloaded the worker. Chat swaps cancel the in-flight stream
        and reset services here so the new chat model takes over cleanly. Works
        regardless of which container the button is mounted in.
        """
        self._refresh()
        if self._scope != "chat":
            return
        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = self.app.screen
        if isinstance(screen, ChatScreen):
            screen.apply_model_change()
        else:
            reset_services()


class ChatModePill(Static, can_focus=True):
    """Single focusable mode pill; Enter / Space picks this pill's mode."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "select", "Pick mode", show=False),
        Binding("space", "select", "Pick mode", show=False),
    ]

    def action_select(self) -> None:
        toggle = next(
            (n for n in self.ancestors_with_self if isinstance(n, ChatModeToggle)),
            None,
        )
        if toggle is None:
            return
        target = (
            ChatMode.SEARCH.value if self.id == _CHAT_MODE_SEARCH_PILL_ID else ChatMode.CHAT.value
        )
        toggle._set_mode(target)


class ChatModeToggle(Widget, can_focus=False):
    """Two-pill control toggling cfg.chat_mode between Search and Chat.

    The toggle itself is not focusable; the inner pills are. Tab walks
    Search then Chat, Enter / Space picks. The container keeps left /
    right arrow handling so the legacy keyboard flow still works.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("left", "select_search", "Search mode", show=False),
        Binding("right", "select_chat", "Chat mode", show=False),
    ]

    def __init__(self) -> None:
        super().__init__(id=_CHAT_MODE_TOGGLE_ID)

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield ChatModePill(
                msg.CHAT_MODE_SEARCH_LABEL,
                id=_CHAT_MODE_SEARCH_PILL_ID,
                classes=_CHAT_MODE_PILL_CLASS,
            )
            yield ChatModePill(
                msg.CHAT_MODE_CHAT_LABEL,
                id=_CHAT_MODE_CHAT_PILL_ID,
                classes=_CHAT_MODE_PILL_CLASS,
            )

    def on_mount(self) -> None:
        self._refresh()

    def refresh_state(self) -> None:
        """Repaint label/state. Call after settings or embedding-model changes."""
        if self.is_mounted:
            self._refresh()

    def _embedding_ready(self) -> bool:
        return is_model_available(cfg.embedding_model, get_services().provider)

    def _refresh(self) -> None:
        ready = self._embedding_ready()
        mode = cfg.chat_mode if ready else ChatMode.CHAT.value
        active_search = mode == ChatMode.SEARCH.value
        search_pill = self.query_one(f"#{_CHAT_MODE_SEARCH_PILL_ID}", ChatModePill)
        chat_pill = self.query_one(f"#{_CHAT_MODE_CHAT_PILL_ID}", ChatModePill)
        # Search half is disabled whenever embedding isn't ready; Chat is
        # always reachable so it never carries the disabled class.
        search_pill.set_class(active_search, _CHAT_MODE_ACTIVE_CLASS)
        search_pill.set_class(not ready, _CHAT_MODE_DISABLED_CLASS)
        chat_pill.set_class(not active_search, _CHAT_MODE_ACTIVE_CLASS)
        chat_pill.set_class(False, _CHAT_MODE_DISABLED_CLASS)
        # Parent carries the disabled class so external selectors can
        # disable interaction on the whole toggle when search is gated.
        self.set_class(not ready, _CHAT_MODE_DISABLED_CLASS)
        self.tooltip = (
            msg.CHAT_MODE_TOGGLE_DISABLED_TOOLTIP if not ready else msg.CHAT_MODE_TOGGLE_TOOLTIP
        )

    def _set_mode(self, target: str) -> bool:
        """Apply *target* if it differs from the current mode and Search is allowed."""
        if cfg.chat_mode == target:
            return False
        if target == ChatMode.SEARCH.value and not self._embedding_ready():
            return False
        apply_setting(self.app, "chat_mode", target)
        self._refresh()
        return True

    def toggle(self) -> bool:
        """Flip mode if embedding is ready. Returns True when the mode changed."""
        target = (
            ChatMode.CHAT.value if cfg.chat_mode == ChatMode.SEARCH.value else ChatMode.SEARCH.value
        )
        return self._set_mode(target)

    def on_click(self, event: events.Click) -> None:
        event.stop()
        # Click on a specific pill picks that side; click on the container
        # frame falls through to a toggle.
        widget = event.widget
        if widget is not None:
            wid = widget.id
            if wid == _CHAT_MODE_SEARCH_PILL_ID:
                self._set_mode(ChatMode.SEARCH.value)
                return
            if wid == _CHAT_MODE_CHAT_PILL_ID:
                self._set_mode(ChatMode.CHAT.value)
                return
        self.toggle()

    def action_flip_mode(self) -> None:
        self.toggle()

    def action_select_search(self) -> None:
        self._set_mode(ChatMode.SEARCH.value)

    def action_select_chat(self) -> None:
        self._set_mode(ChatMode.CHAT.value)


class RoleRow(Widget, can_focus=False):
    """One role unit in the bar: a colored role pill + its picker button."""

    def __init__(self, *, scope: PickerScope) -> None:
        super().__init__()
        self.scope: PickerScope = scope
        self._key: str = config_key_for_scope(scope)

    def compose(self) -> ComposeResult:
        yield Static("", classes="model-bar-pill")
        yield ModelPickerButton(scope=self.scope, button_id=f"model-pick-{self.scope}")

    def on_mount(self) -> None:
        self.refresh_state()

    @property
    def is_active(self) -> bool:
        return bool(getattr(cfg, self._key))

    def _is_nullable(self) -> bool:
        from lilbee.app.settings_map import SETTINGS_MAP

        defn = SETTINGS_MAP.get(self._key)
        return defn is not None and defn.nullable

    def on_click(self, event: events.Click) -> None:
        """Click the pill to toggle an optional role off; otherwise open the picker.

        The picker button stops its own click events, so this handler only runs
        for clicks on the role pill (or the row gutter).
        """
        event.stop()
        if self._is_nullable() and self.is_active:
            apply_model_pick(self, key=self._key, ref="", on_done=self.refresh_state)
        else:
            self.query_one(ModelPickerButton).open_picker()

    def refresh_state(self) -> None:
        """Repaint the role pill (colored when on, muted when off) and the picker label."""
        active = self.is_active
        self.set_class(active, "-active")
        self.set_class(not active, "-off")
        bg, fg = _SCOPE_PILL_COLORS[self.scope] if active else _OFF_PILL_COLORS
        # Tolerate "children not mounted yet" only; a real pill/repaint failure
        # should surface rather than be silently swallowed.
        with contextlib.suppress(NoMatches):
            self.query_one(".model-bar-pill", Static).update(
                pill(_SCOPE_TO_LABEL[self.scope], bg, fg)
            )
            self.query_one(ModelPickerButton).repaint()


class ModelBar(Widget, can_focus=False):
    """Horizontal band of the four role pickers + the Search/Chat toggle, below the input."""

    app: LilbeeApp  # type: ignore[assignment]
    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    _SCOPES: ClassVar[tuple[PickerScope, ...]] = ("chat", "embed", "vision", "rerank")

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._options_cache: dict[str, tuple[tuple[str, str], ...]] = {}

    def compose(self) -> ComposeResult:
        with Horizontal(classes="model-bar-roles"):
            for scope in self._SCOPES:
                yield RoleRow(scope=scope)
            yield ChatModeToggle()
        yield Static("", id=_CLOUD_WARNING_ID, classes="cloud-warning")

    def on_mount(self) -> None:
        self._refresh_cloud_warning()
        self._scan_models()
        self.app.settings_changed_signal.subscribe(self, self._on_settings_changed)

    def _on_settings_changed(self, payload: tuple[str, object]) -> None:
        key, _ = payload
        scope = model_field_to_picker_scope().get(key)
        if scope is None:
            return
        for row in self.query(RoleRow):
            if row.scope == scope:
                row.refresh_state()
        if key == "chat_model":
            self._refresh_cloud_warning()

    @work(thread=True)
    def _scan_models(self) -> None:
        """Scan installed models off the UI thread and populate every role button."""
        buckets = classify_installed_models_full()
        scope_to_options: dict[str, list[ModelOption]] = {
            "chat": list(buckets.get(ModelTask.CHAT, [])),
            "embed": list(buckets.get(ModelTask.EMBEDDING, [])),
            "vision": list(buckets.get(ModelTask.VISION, [])),
            "rerank": list(buckets.get(ModelTask.RERANK, [])),
        }
        call_from_thread(self, self._populate, scope_to_options)

    def _populate(self, scope_to_options: dict[str, list[ModelOption]]) -> None:
        for row in self.query(RoleRow):
            # Empty pool stays empty: the picker shows just its "Browse catalog"
            # row rather than a pickable "(none)" pseudo-model.
            opts = scope_to_options.get(row.scope, [])
            fingerprint = tuple((o.label, o.ref) for o in opts)
            if self._options_cache.get(row.scope) != fingerprint:
                row.query_one(ModelPickerButton).set_options(opts)
                self._options_cache[row.scope] = fingerprint
        self._refresh_cloud_warning()

    def _refresh_cloud_warning(self) -> None:
        """Show a warning if the active chat model routes to a cloud provider."""
        warning = self.query_one(f"#{_CLOUD_WARNING_ID}", Static)
        label = _cloud_provider_label(cfg.chat_model)
        if label is None:
            warning.remove_class("-visible")
            return
        warning.update(msg.MODEL_BAR_CLOUD_PROVIDER_WARNING.format(provider=label))
        warning.add_class("-visible")

    def refresh_models(self) -> None:
        """Re-scan installed models (called after downloads complete)."""
        self._scan_models()
