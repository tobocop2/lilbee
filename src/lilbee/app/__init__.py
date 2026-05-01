"""Surface-agnostic use-case orchestration shared by CLI, HTTP, MCP, and TUI.

Each module here is a thin orchestration layer around the underlying domain
services. Surfaces import the use-case directly; this package exposes nothing
through ``__init__`` so import paths stay obvious and grep-friendly.
"""
