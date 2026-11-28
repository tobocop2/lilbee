"""The TUI click census: walk every surface, open it, and compare a click with the key.

A surface is a `Screen` subclass that a `lilbee.cli.tui` module defines. The oracle is the key:
what Enter, or a binding that a surface shows, does is what a click must do.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import pkgutil
import sys
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

from textual.app import App, ComposeResult
from textual.errors import NoWidget
from textual.message_pump import MessagePump
from textual.pilot import Pilot
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Input,
    ListView,
    OptionList,
    Select,
    Static,
    Switch,
    TabbedContent,
    TextArea,
)
from textual.widgets._toggle_button import ToggleButton

from lilbee.cli.tui.widgets.task_bar import TaskBar
from lilbee.core.config import cfg
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost

SIZES = [(80, 24), (120, 40)]
SIZE_IDS = [f"{w}x{h}" for w, h in SIZES]
# Messages of the framework itself (input, layout, lifecycle) are not an effect of a control.
NOISE_MODULES = ("textual.events", "textual.messages")
FOCUS_NOISE = ("Blurred", "Focused")

Effects = set[str]
# Lists and tables: a click on a row chooses it, and a choice also closes a picker.
CHOICE_WIDGETS = (OptionList, DataTable, ListView)


class NoBuilder(Exception):
    """A surface needs constructor arguments and the census has no builder for it."""


def _subclasses(cls: type) -> set[type]:
    out: set[type] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _subclasses(sub)
    return out


def defined_by_its_module(cls: type) -> bool:
    """True for a class that a `class` statement in a lilbee TUI module made.

    Another test builds subclasses with `type()` and a borrowed module name; those have no source.
    """
    module = sys.modules[cls.__module__]
    return (
        cls.__module__.startswith("lilbee.cli.tui.") and getattr(module, cls.__name__, None) is cls
    )


def import_every_tui_module() -> None:
    import lilbee.cli.tui as tui

    for mod in pkgutil.walk_packages(tui.__path__, "lilbee.cli.tui."):
        importlib.import_module(mod.name)


def discover_surfaces() -> list[type]:
    """Every screen class that the TUI package defines, found by walking the class tree."""
    from lilbee.cli.tui.widgets.dialog import Dialog

    import_every_tui_module()
    return sorted(
        (c for c in _subclasses(Screen) if defined_by_its_module(c) and c is not Dialog),
        key=lambda c: c.__name__,
    )


def builders() -> dict[type, Callable[[], Screen[Any]]]:
    """One builder for each surface whose constructor needs arguments, keyed by class."""
    from lilbee.app.engine_service import EngineOption
    from lilbee.catalog.types import ModelFormat
    from lilbee.cli.tui.format_rows import FormatRow
    from lilbee.cli.tui.screens.analyze_report import AnalyzeReportScreen
    from lilbee.cli.tui.screens.engine_picker import EnginePickerModal
    from lilbee.cli.tui.screens.model_info import ModelInfoModal
    from lilbee.cli.tui.screens.model_picker import ModelPickerModal
    from lilbee.cli.tui.screens.profile_dialogs import (
        ApplyProfileDialog,
        ProfilePathDialog,
        SaveProfileDialog,
    )
    from lilbee.cli.tui.screens.profile_library import ProfileLibrary
    from lilbee.cli.tui.screens.task_detail import TaskDetailModal
    from lilbee.cli.tui.task_queue import TaskReport
    from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
    from lilbee.cli.tui.widgets.fork_picker import ForkPicker, ForkPoint
    from lilbee.cli.tui.widgets.format_picker import FormatPicker
    from lilbee.cli.tui.widgets.notice_dialog import NoticeDialog
    from lilbee.core.profile_files import ProfileCatalog, ProfileFolder
    from tests.test_tui_analyze import _report
    from tests.test_tui_small_fixes import _apply_dialog, _info_dialog, _picker_rows

    return {
        AnalyzeReportScreen: lambda: AnalyzeReportScreen(_report(), "~/archive/city-records"),
        ApplyProfileDialog: _apply_dialog,
        ConfirmDialog: lambda: ConfirmDialog("Remove model", "Remove this model?"),
        EnginePickerModal: lambda: EnginePickerModal(
            "Engine",
            "For this model",
            "org/m/f.gguf",
            None,
            [EngineOption("auto"), EngineOption("llama-cpp")],
        ),
        ForkPicker: lambda: ForkPicker([ForkPoint(2, "Answer", "Question")]),
        FormatPicker: lambda: FormatPicker(
            "Gemma", [FormatRow(ModelFormat.GGUF, True, None, None, None)]
        ),
        ModelInfoModal: _info_dialog,
        ModelPickerModal: lambda: ModelPickerModal(
            scope="chat", options=_picker_rows(6), pick_later=lambda ref: None
        ),
        NoticeDialog: lambda: NoticeDialog("Heads up", "A short notice."),
        ProfileLibrary: lambda: ProfileLibrary(on_change=lambda: None),
        ProfilePathDialog: lambda: ProfilePathDialog(
            title="Import a profile",
            explain="Pick a file.",
            action_label="Import",
            write=lambda request: None,
            folders=(ProfileFolder.GLOBAL, ProfileFolder.PROJECT),
        ),
        SaveProfileDialog: lambda: SaveProfileDialog(
            title="Save a profile",
            explain="Name it.",
            catalog=ProfileCatalog(entries=()),
            folders=(ProfileFolder.GLOBAL,),
            initial="",
        ),
        TaskDetailModal: lambda: TaskDetailModal(
            "ingest", TaskReport(failed={"a.pdf": "unreadable"})
        ),
    }


def _needs_args(cls: type) -> bool:
    params = list(inspect.signature(cls.__init__).parameters.values())[1:]
    return any(
        p.default is p.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        for p in params
    )


def build(cls: type) -> Screen[Any]:
    """Make one instance of a surface: its builder, or no arguments when it needs none."""
    made = builders().get(cls)
    if made is not None:
        return made()
    if _needs_args(cls):
        raise NoBuilder(f"{cls.__name__} needs arguments and the census has no builder for it")
    instance: Screen[Any] = cls()
    return instance


class SurfaceApp(LilbeeAppHost):
    """The app host with one surface pushed over the Chat view."""

    def __init__(self, cls: type) -> None:
        super().__init__()
        self._cls = cls
        self.opened: Screen[Any] | None = None

    def compose(self) -> ComposeResult:
        yield Static("")

    def on_mount(self) -> None:
        from lilbee.cli.tui.app import _CHAT_SCREEN_NAME
        from lilbee.cli.tui.screens.chat import ChatScreen

        # The host skips the install of the Chat view; a view tab switches to it by name.
        if self._cls is not ChatScreen:
            self.install_screen(ChatScreen(), name=_CHAT_SCREEN_NAME)
        self.opened = build(self._cls)
        self.push_screen(self.opened)


async def open_surface(pilot: Pilot[Any], app: SurfaceApp) -> Screen[Any]:
    await pilot.pause()
    assert await wait_until(
        pilot, lambda: app.opened is not None and app.screen is app.opened, max_pauses=40
    )
    await settle(pilot)
    return app.screen


def label(widget: Widget) -> str:
    return f"{type(widget).__name__}#{widget.id}" if widget.id else type(widget).__name__


def _widget_state(w: Widget) -> tuple[object, ...]:
    """What a person sees of one widget: classes, value, label and text."""
    value: object = ""
    if isinstance(w, (Input, Switch, ToggleButton, Select)):
        value = w.value
    elif isinstance(w, TabbedContent):
        value = w.active
    text = ""
    if isinstance(w, Static):
        text = str(w.content)[:80]
    elif isinstance(w, (Button, ToggleButton)):
        text = str(w.label)
    return (
        type(w).__name__,
        w.id,
        tuple(sorted(c for c in w.classes if c != "-hover")),
        str(value),
        text,
    )


def _screen_state(screen: Screen[Any]) -> list[tuple[object, ...]]:
    """The state of every widget that a control can change.

    The task bar is left out: it polls the task queue on its own timer, so it shows a
    task that a control started at a time that no Textual signal reports. The Footer is
    left out: it rebuilds its keys after any focus change, later than any Textual wait.
    """
    return [
        _widget_state(w)
        for w in screen.query("*")
        if not any(isinstance(a, (TaskBar, Footer)) for a in w.ancestors_with_self)
    ]


class Recorder:
    """Records the effects of one input: messages, the screen stack, visible state, focus."""

    def __init__(self, app: App[Any], *, track_focus: bool = False) -> None:
        self.app = app
        self.track_focus = track_focus
        self.effects: Effects = set()
        self._real_post = MessagePump.post_message

    def __enter__(self) -> Recorder:
        self.stack_before = [type(s).__name__ for s in self.app.screen_stack]
        self.screen_before = self.app.screen
        self.focus_before = self.app.focused
        self.state_before = _screen_state(self.app.screen)
        real_post = self._real_post
        effects = self.effects

        def post(pump: MessagePump, message: Any) -> Any:
            kind = type(message)
            if not kind.__module__.startswith(NOISE_MODULES) and kind.__name__ not in FOCUS_NOISE:
                effects.add(f"msg:{kind.__qualname__}")
            return real_post(pump, message)

        MessagePump.post_message = post  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: object) -> None:
        MessagePump.post_message = self._real_post  # type: ignore[method-assign]
        after = [type(s).__name__ for s in self.app.screen_stack]
        if len(after) < len(self.stack_before):
            self.effects.add("stack:pop")
        elif len(after) > len(self.stack_before):
            self.effects.add(f"stack:push:{after[-1]}")
        elif after != self.stack_before:
            self.effects.add(f"stack:swap:{after[-1]}")
        elif (
            self.app.screen is self.screen_before
            and _screen_state(self.app.screen) != self.state_before
        ):
            self.effects.add("dom:changed")
        focused = self.app.focused
        if self.track_focus and focused is not None and focused is not self.focus_before:
            self.effects.add(f"focus:{label(focused)}")


async def settle(pilot: Pilot[Any]) -> None:
    """Wait on Textual's own signals: every animation, every worker, then the message queues.

    A timer that fires later than these signals report is not waited for: Textual has no signal
    for it, and a control whose effect hangs on one needs a test of its own.
    """
    await pilot.wait_for_scheduled_animations()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


def owns_enter(app: SurfaceApp, widget: Widget) -> bool:
    """True when the widget itself binds Enter or Space, not its screen."""
    active = app.active_bindings
    return any(key in active and active[key].node is widget for key in ("enter", "space"))


def is_text_entry(widget: Widget) -> bool:
    """A click on text entry places the cursor; there is no action to compare."""
    return isinstance(widget, (Input, TextArea))


def _option_row(widget: OptionList) -> int:
    """The row of the option that Enter would act on, found by drawing the rows."""
    target = widget.highlighted
    if target is None:
        target = next(
            i for i in range(widget.option_count) if not widget.get_option_at_index(i).disabled
        )
    for y in range(widget.size.height):
        for segment in widget.render_line(y):
            meta = segment.style.meta if segment.style is not None else {}
            if meta.get("option") == target:
                return y
    return 0


def _clicks_to_act(widget: Widget) -> int:
    """A card grid highlights on one click and acts on a second click on the same card.

    The grid refuses a one-click install on purpose: a mis-tap must never start a download.
    """
    from lilbee.cli.tui.widgets.model_grid import ModelGrid

    return 2 if isinstance(widget, ModelGrid) else 1


def _table_row(widget: DataTable[Any]) -> int:
    """The line of the cursor row that Enter would act on, counted from the top of the content."""
    header = widget.header_height if widget.show_header else 0
    above = sum(row.height for row in widget.ordered_rows[: widget.cursor_row])
    return header + above - widget.scroll_offset.y


def _aim(widget: Widget) -> tuple[int, int]:
    """The click offset, from the top-left of the widget, of the place where Enter acts."""
    # The rows of a list or a table start inside the border box that the offset counts from.
    inner = widget.content_region.offset - widget.region.offset
    if isinstance(widget, OptionList):
        return (inner.x + 2, inner.y + _option_row(widget))
    if isinstance(widget, DataTable):
        return (inner.x + 1, inner.y + _table_row(widget))
    return (0, 0)


def _assert_aim_on(widget: Widget, offset: tuple[int, int]) -> None:
    """Fail, naming the widget hit, when the point is not on the widget or a child of it."""
    x, y = widget.region.offset + offset
    try:
        hit = widget.screen.get_widget_at(x, y)[0]
    except NoWidget:
        raise AssertionError(f"a click at {(x, y)} hits nothing, not {label(widget)}") from None
    if hit is not widget and widget not in hit.ancestors:
        raise AssertionError(f"a click at {(x, y)} hits {label(hit)}, not {label(widget)}")


async def click(pilot: Pilot[Any], widget: Widget) -> None:
    """Click the place where Enter acts: the highlighted option, the cursor row, or the top-left."""
    widget.scroll_visible(animate=False)
    await settle(pilot)
    offset = _aim(widget)
    _assert_aim_on(widget, offset)
    await pilot.click(widget, offset=offset, times=_clicks_to_act(widget))


@contextlib.contextmanager
def settings_restored() -> Iterator[None]:
    """Put every setting back after a probe, so a click starts from the same state as Enter."""
    snapshot = cfg.model_copy()
    try:
        yield
    finally:
        for name in type(cfg).model_fields:
            setattr(cfg, name, getattr(snapshot, name))


async def with_surface(
    cls: type,
    size: tuple[int, int],
    act: Callable[[Pilot[Any], SurfaceApp, Screen[Any]], Awaitable[Effects]],
) -> Effects:
    """Open the surface afresh, run `act` on it, return the effects it recorded."""
    app = SurfaceApp(cls)
    with settings_restored():
        async with app.run_test(size=size) as pilot:
            screen = await open_surface(pilot, app)
            return await act(pilot, app, screen)


async def focus_labels(cls: type, size: tuple[int, int]) -> list[str]:
    app = SurfaceApp(cls)
    async with app.run_test(size=size) as pilot:
        screen = await open_surface(pilot, app)
        return [label(w) for w in screen.focus_chain]


class WidgetProbe:
    """The effects of Enter and of a click on one focusable widget, each on a fresh surface."""

    def __init__(self, name: str, owns_enter: bool, text_entry: bool, effects: Effects) -> None:
        self.name = name
        self.owns_enter = owns_enter
        self.text_entry = text_entry
        self.effects = effects


async def probe_widget(cls: type, size: tuple[int, int], index: int, how: str) -> WidgetProbe:
    app = SurfaceApp(cls)
    with settings_restored():
        async with app.run_test(size=size) as pilot:
            screen = await open_surface(pilot, app)
            widget = screen.focus_chain[index]
            widget.focus()
            await settle(pilot)
            owns = owns_enter(app, widget)
            entry = is_text_entry(widget)
            effects: Effects = set()
            if owns and not entry:
                with Recorder(app) as rec:
                    if how == "enter":
                        await pilot.press("enter")
                    else:
                        await click(pilot, widget)
                    await settle(pilot)
                effects = rec.effects
            return WidgetProbe(label(widget), owns, entry, effects)


def is_reached(screen: Screen[Any], widget: Widget) -> bool:
    """True when a click at the centre of the widget lands on it or on a child of it."""
    region = widget.region
    if region.width <= 0 or region.height <= 0:
        return False
    x, y = region.x + region.width // 2, region.y + region.height // 2
    try:
        target = screen.get_widget_at(x, y)[0]
    except NoWidget:
        return False
    return target is widget or widget in target.ancestors


async def key_effects(cls: type, size: tuple[int, int], key: str) -> Effects:
    """The effects of one key, pressed with the last focusable widget focused."""

    async def act(pilot: Pilot[Any], app: SurfaceApp, screen: Screen[Any]) -> Effects:
        if screen.focus_chain:
            screen.focus_chain[-1].focus()
            await settle(pilot)
        with Recorder(app, track_focus=True) as rec:
            await pilot.press(key)
            await settle(pilot)
        return rec.effects

    return await with_surface(cls, size, act)


async def backdrop_click_effects(cls: type, size: tuple[int, int]) -> Effects:
    """The effects of a click on the top-left corner, which is outside any dialog box."""

    async def act(pilot: Pilot[Any], app: SurfaceApp, screen: Screen[Any]) -> Effects:
        with Recorder(app, track_focus=True) as rec:
            await pilot.click(offset=(0, 0))
            await settle(pilot)
        return rec.effects

    return await with_surface(cls, size, act)


async def mouse_close_controls(cls: type, size: tuple[int, int]) -> list[str]:
    """The widgets whose click closes the dialog and that are not a list of choices.

    Choosing a list row also closes a picker, so a row is not a way to leave it with no choice.
    A click outside the box counts too, and shows as `backdrop`.
    """
    names = await focus_labels(cls, size)
    found: list[str] = []
    for index, name in enumerate(names):

        async def act(
            pilot: Pilot[Any], app: SurfaceApp, screen: Screen[Any], index: int = index
        ) -> Effects:
            widget = screen.focus_chain[index]
            if isinstance(widget, CHOICE_WIDGETS):
                return set()
            with Recorder(app) as rec:
                await click(pilot, widget)
                await settle(pilot)
            return rec.effects

        if "stack:pop" in await with_surface(cls, size, act):
            found.append(name)
    # Textual's own palette closes on a click outside its box; the ruling counts that.
    if "stack:pop" in await backdrop_click_effects(cls, size):
        found.append("backdrop")
    return found


async def click_effects(cls: type, size: tuple[int, int], index: int) -> Effects:
    """The effects of a click on one widget, with another widget focused first."""

    async def act(pilot: Pilot[Any], app: SurfaceApp, screen: Screen[Any]) -> Effects:
        chain = screen.focus_chain
        widget = chain[index]
        other = chain[-1] if widget is not chain[-1] else chain[0]
        other.focus()
        await settle(pilot)
        with Recorder(app, track_focus=True) as rec:
            await click(pilot, widget)
            await settle(pilot)
        return rec.effects

    return await with_surface(cls, size, act)
