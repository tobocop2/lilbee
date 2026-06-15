"""Nuitka plugin: ship Playwright's `node` driver byte-for-byte.

Nuitka treats ``playwright/driver/node`` as a relocatable binary and rewrites
its RPATH with ``patchelf``. patchelf corrupts that ~121 MB binary (it has many
ELF notes), so the bundled node segfaults at startup and ``lilbee setup
crawler`` dies with SIGSEGV. node needs no RPATH -- it resolves its libraries
from the system loader -- so the rewrite is both unnecessary and harmful.

This restores the pristine node bytes after Nuitka has copied and patched it,
keeping node's executable entry-point handling (permissions, onefile packaging)
untouched. It removes the dependency on the build image's patchelf version.
"""

import os
import shutil

from nuitka.plugins.PluginBase import NuitkaPluginBase

_NODE_REL_PATH = "playwright/driver/node"


class NuitkaPluginPlaywrightNodeVerbatim(NuitkaPluginBase):
    plugin_name = "playwright-node-verbatim"
    plugin_desc = "Restore Playwright's node driver to its pristine, unpatched bytes."

    @staticmethod
    def isAlwaysEnabled():
        return True

    def _pristine_node(self):
        playwright_path = self.locateModule("playwright")
        if not playwright_path:
            return None
        if os.path.isfile(playwright_path):  # __init__.py rather than the package dir
            playwright_path = os.path.dirname(playwright_path)
        node = os.path.join(playwright_path, "driver", "node")
        return node if os.path.isfile(node) else None

    def onCopiedDLL(self, dll_filename):
        if not dll_filename.replace(os.sep, "/").endswith(_NODE_REL_PATH):
            return
        pristine = self._pristine_node()
        if pristine is None:
            self.sysexit("playwright-node-verbatim: could not locate the pristine node driver")
        mode = os.stat(dll_filename).st_mode
        shutil.copyfile(pristine, dll_filename)
        os.chmod(dll_filename, mode)
        self.info("Restored Playwright node driver verbatim: %s" % dll_filename)
