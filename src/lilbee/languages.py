"""File extension to tree-sitter language mapping.

Maps file extensions to language names recognized by
tree-sitter-language-pack's process() API. The full list of
supported languages is available via manifest_languages().

This map only includes languages with common file extensions.
Languages without entries here fall back to token-based chunking.
"""

EXT_TO_LANG: dict[str, str] = {
    # Systems / compiled
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".d": "d",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".rs": "rust",
    ".scala": "scala",
    ".swift": "swift",
    ".zig": "zig",
    ".nim": "nim",
    # Scripting / dynamic
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rb": "ruby",
    ".php": "php",
    ".lua": "lua",
    ".pl": "perl",
    ".r": "r",
    ".R": "r",
    ".jl": "julia",
    ".ex": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".dart": "dart",
    ".gd": "gdscript",
    # Shell
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    # Functional
    ".clj": "clojure",
    ".cljs": "clojure",
    ".elm": "elm",
    ".fs": "fsharp",
    ".fsx": "fsharp",
    ".groovy": "groovy",
    ".gradle": "groovy",
    # Web / markup
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".vue": "vue",
    ".svelte": "svelte",
    # Data / config / infra
    ".sql": "sql",
    ".sol": "solidity",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".dockerfile": "dockerfile",
    ".ps1": "powershell",
    ".psm1": "powershell",
}
