"""Rich rendering helpers. `cli` is the only layer that touches the terminal (CLAUDE.md 3.1)."""

from __future__ import annotations

from urllib.parse import quote

from rich.console import Console
from rich.panel import Panel

from obsidian_local_rag.domain.models import Citation

console = Console()


def print_error(message: str, *, debug: bool = False) -> None:
    """Show `message` as a concise error panel. With `debug=True`, also print the traceback.

    Must be called from within an `except` block when `debug=True` (it reads the current
    exception via `rich.Console.print_exception`).
    """
    console.print(Panel(message, title="Error", border_style="red"))
    if debug:
        console.print_exception()


def print_not_implemented(command_name: str, *, debug: bool = False) -> None:
    """Show a friendly "not implemented yet" panel for a stubbed command/service call."""
    console.print(
        Panel(
            f"`{command_name}` is not implemented yet.",
            title="Not implemented",
            border_style="yellow",
        )
    )
    if debug:
        console.print_exception()


def render_sources_footer(citations: dict[str, Citation], vault_dir_basename: str) -> str:
    """Render the sources footer: `[S3] relative/path.md:L10-L42` plus a clickable
    `obsidian://open?vault=...&file=...` hyperlink, both URL-encoded.
    """
    encoded_vault = quote(vault_dir_basename)
    lines: list[str] = []
    for key in sorted(citations):
        citation = citations[key]
        encoded_file = quote(str(citation.path))
        link = f"obsidian://open?vault={encoded_vault}&file={encoded_file}"
        lines.append(
            f"[{key}] {citation.path}:L{citation.start_line}-L{citation.end_line} ({link})"
        )
    return "\n".join(lines)
