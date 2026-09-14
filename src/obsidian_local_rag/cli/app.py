"""Typer CLI entrypoint: registers all `obsidian-rag` subcommands."""

from __future__ import annotations

import typer

from obsidian_local_rag.cli.commands import chat_cmd, graph_info_cmd, index_cmd, stats_cmd

app = typer.Typer(name="obsidian-rag", help="Agentic RAG assistant over a local Obsidian vault.")

app.command(name="index")(index_cmd.index)
app.command(name="chat")(chat_cmd.chat)
app.command(name="stats")(stats_cmd.stats)
app.command(name="graph-info")(graph_info_cmd.graph_info)


if __name__ == "__main__":
    app()
