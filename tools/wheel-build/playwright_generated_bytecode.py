"""Nuitka plugin: include Playwright's generated API modules as bytecode, not C."""

from nuitka.plugins.PluginBase import NuitkaPluginBase

_BYTECODE_MODULES = frozenset(
    {
        "playwright.async_api._generated",
        "playwright.sync_api._generated",
    }
)


class NuitkaPluginPlaywrightGeneratedBytecode(NuitkaPluginBase):
    plugin_name = "playwright-generated-bytecode"
    plugin_desc = "Include Playwright's generated API modules as bytecode, not C."

    @staticmethod
    def isAlwaysEnabled():
        return True

    def decideCompilation(self, module_name):
        if module_name in _BYTECODE_MODULES:
            return "bytecode"
        return None
