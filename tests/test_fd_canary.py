"""Canary for the closed-descriptor trace: one owner locks a file, another closes its descriptor.

Branch diag/fd-closer only. The trace summary must print FDC-FOREIGN for each run of this test,
with the stack of the closing thread and of the owner that took the lock.
"""

import errno
import fcntl
import os
import threading
from pathlib import Path

import pytest


@pytest.mark.parametrize("copy", range(4))
def test_second_owner_closes_the_lock_descriptor(tmp_path: Path, copy: int) -> None:
    path = tmp_path / f"fd-canary-{os.getpid()}.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    closer = threading.Thread(target=os.close, args=(fd,), name="fd-canary-closer")
    closer.start()
    closer.join()
    with pytest.raises(OSError) as raised:
        fcntl.flock(fd, fcntl.LOCK_UN)
    assert raised.value.errno == errno.EBADF
