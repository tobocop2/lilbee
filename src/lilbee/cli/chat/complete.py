"""Tab completion for the chat REPL."""

from __future__ import annotations

from lilbee.config import cfg

_ADD_PREFIX = "/add "
_MODEL_PREFIX = "/model "
_VISION_PREFIX = "/vision "
_SET_PREFIX = "/set "


def list_installed_models(*, exclude_vision: bool = False) -> list[str]:
    """Return installed model names with explicit tags, excluding embedding models.

    When *exclude_vision* is True, also filters out known vision catalog models.
    """
    from lilbee.providers import get_provider

    try:
        provider = get_provider()
        embed_base = cfg.embedding_model.split(":")[0]
        models = [m for m in provider.list_models() if m.split(":")[0] != embed_base]
        if exclude_vision:
            from lilbee.models import VISION_CATALOG

            vision_names = {m.name for m in VISION_CATALOG}
            models = [m for m in models if m not in vision_names]
        return models
    except Exception:
        return []


def make_completer():  # type: ignore[no-untyped-def]
    """Build a completer class that inherits from prompt_toolkit.completion.Completer."""
    from prompt_toolkit.completion import Completer, Completion, PathCompleter
    from prompt_toolkit.document import Document

    class LilbeeCompleter(Completer):
        def get_completions(self, document, complete_event):  # type: ignore[no-untyped-def,override]
            from lilbee.cli.chat.slash import _SETTINGS_MAP, _SLASH_COMMANDS

            text = document.text_before_cursor
            if text.startswith(_ADD_PREFIX):
                sub_text = text[len(_ADD_PREFIX) :]
                sub_doc = Document(sub_text, len(sub_text))
                yield from PathCompleter(expanduser=True).get_completions(sub_doc, complete_event)
            elif text.startswith(_MODEL_PREFIX):
                prefix = text[len(_MODEL_PREFIX) :]
                for name in list_installed_models(exclude_vision=True):
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))
            elif text.startswith(_SET_PREFIX):
                prefix = text[len(_SET_PREFIX) :]
                for name in _SETTINGS_MAP:
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))
            elif text.startswith(_VISION_PREFIX):
                from lilbee.models import VISION_CATALOG

                prefix = text[len(_VISION_PREFIX) :]
                if "off".startswith(prefix):
                    yield Completion("off", start_position=-len(prefix))
                for model in VISION_CATALOG:
                    if model.name.startswith(prefix):
                        yield Completion(model.name, start_position=-len(prefix))
            elif text.startswith("/"):
                prefix = text[1:]
                for cmd in _SLASH_COMMANDS:
                    if cmd.startswith(prefix):
                        yield Completion(f"/{cmd}", start_position=-len(text))

    return LilbeeCompleter()
