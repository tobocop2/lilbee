"""Interactive chat mode — REPL, slash commands, tab completion, background sync."""

from lilbee.cli.chat.complete import list_ollama_models, make_completer
from lilbee.cli.chat.loop import chat_loop, sync_toolbar
from lilbee.cli.chat.slash import QuitChat, dispatch_slash
from lilbee.cli.chat.stream import stream_response
from lilbee.cli.chat.sync import SyncStatus, run_sync_background, shutdown_executor

__all__ = [
    "QuitChat",
    "SyncStatus",
    "chat_loop",
    "dispatch_slash",
    "list_ollama_models",
    "make_completer",
    "run_sync_background",
    "shutdown_executor",
    "stream_response",
    "sync_toolbar",
]
