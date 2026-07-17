"""Task 4: deduplicate. Both functions must keep passing their tests."""


def render_user_row(name: str, age: int) -> str:
    cleaned = name.strip().title()
    if not cleaned:
        cleaned = "Unknown"
    label = f"{cleaned} ({age})"
    return f"| {label:<30} |"


def render_admin_row(name: str, age: int) -> str:
    cleaned = name.strip().title()
    if not cleaned:
        cleaned = "Unknown"
    label = f"{cleaned} ({age}) [admin]"
    return f"| {label:<30} |"
