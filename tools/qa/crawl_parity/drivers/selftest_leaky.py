"""A driver that leaves a process, a listening socket, a temp directory and a thread, on purpose.

The left-behind self-test runs it through the same code that runs a real driver.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading

from _driver_io import PageOut, RunOut, crawl_arguments, now, thread_count, write_pages, write_run

LISTENER = (
    "import socket, time\n"
    "server = socket.socket()\n"
    "server.bind(('127.0.0.1', 0))\n"
    "server.listen()\n"
    "time.sleep(120)\n"
)


def run() -> int:
    """Leave one of each thing behind and report one page."""
    args = crawl_arguments(__doc__ or "")
    record = RunOut(started=now(), threads_before=thread_count())
    record.crawl_started = now()
    subprocess.Popen([sys.executable, "-c", LISTENER], stdin=subprocess.DEVNULL)
    tempfile.mkdtemp(prefix="planted-profile-")
    threading.Thread(target=threading.Event().wait, daemon=True).start()
    record.crawl_ended = now()
    record.threads_after = thread_count()
    write_pages(args.out, [PageOut(args.seed, "planted page", saved_at=now())])
    write_run(args.out, record)
    return 0


if __name__ == "__main__":
    sys.exit(run())
