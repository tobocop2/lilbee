"""CLI command subpackage.

Top-level Typer commands are split across submodules by domain. This
``__init__`` registers each command function on the shared ``app``.
Top-level commands are listed first so ``lilbee --help`` shows them
above the ``setup`` and ``wiki`` sub-typers (attached at the bottom).
"""

from __future__ import annotations

from lilbee.cli.app import app
from lilbee.cli.commands import ingest_sync, meta, search_chat, servers, wiki
from lilbee.cli.commands import setup as setup_module

# Top-level commands listed first so `lilbee --help` shows them before the
# `setup` and `wiki` sub-typers, which read better grouped at the bottom.
app.command()(search_chat.search)
app.command(name="sync")(ingest_sync.sync_cmd)
app.command()(ingest_sync.rebuild)
app.command()(ingest_sync.add)
app.command()(ingest_sync.chunks)
app.command()(ingest_sync.remove)
app.command()(search_chat.ask)
app.command()(search_chat.chat)
app.command()(meta.version)
app.command(name="self-check")(setup_module.self_check_cmd)
app.command(name="self-check-extras")(setup_module.self_check_extras_cmd)
app.command()(meta.status)
app.command()(meta.reset)
app.command()(meta.init)
app.command()(servers.serve)
app.command()(setup_module.token)
app.command()(search_chat.topics)
app.command()(setup_module.login)
app.command(name="mcp")(servers.mcp_cmd)

# Sub-typers come last so `setup` and `wiki` follow the top-level commands
# in `--help`.
app.add_typer(setup_module.setup_app, name="setup")
app.add_typer(wiki.wiki_app, name="wiki")

__all__ = ["app"]
