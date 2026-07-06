# Tool extraction

Lilbee launches llama-server with --jinja, so the server renders each model's own chat template and parses its native tool-call syntax into structured message.tool_calls. A recovery pass in providers/fleet/client.py catches bare-JSON tool calls that models emit as plain content.