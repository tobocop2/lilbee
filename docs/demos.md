# Demos

Terminal recordings showing lilbee in action. Each demo is recorded with [VHS](https://github.com/charmbracelet/vhs) and can be re-recorded with `make demo`.

## Interactive chat

Model switching via tab completion, then a Crown Vic Q&A grounded in an indexed PDF.

![Interactive chat demo](../demos/chat.gif)

## Code search

Add a codebase and search with natural language. Tree-sitter provides AST-aware chunking so results map to real functions and classes.

![Code search demo](../demos/code-search.gif)

## JSON output

Structured JSON output for agents and scripts — pipe through `jq` or parse programmatically.

![JSON output demo](../demos/json.gif)

## Agent integration with opencode

lilbee as a retrieval backend for [opencode](https://github.com/sst/opencode). The agent shells out to `lilbee --json search` to ground its answers in your documents.

![opencode demo](../demos/opencode.gif)
