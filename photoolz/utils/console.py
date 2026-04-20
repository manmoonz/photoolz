from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()


def make_progress(description: str = "Processing", total: int | None = None) -> Progress:
    p = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    return p


def print_photo_table(photos: list[dict], columns: list[str] | None = None) -> None:
    if not photos:
        console.print("[yellow]No photos found.[/yellow]")
        return
    default_cols = ["id", "file_path", "taken_at", "quality_score", "width", "height"]
    cols = columns or default_cols
    table = Table(show_header=True, header_style="bold cyan")
    for col in cols:
        table.add_column(col, no_wrap=(col == "file_path"))
    for p in photos:
        row = []
        for col in cols:
            val = p.get(col) if isinstance(p, dict) else p[col] if col in p.keys() else ""
            if val is None:
                val = ""
            elif isinstance(val, float):
                val = f"{val:.3f}"
            else:
                val = str(val)
            row.append(val)
        table.add_row(*row)
    console.print(table)


def print_album(album: dict) -> None:
    console.print(f"\n[bold green]{album.get('title', 'Untitled')}[/bold green]")
    if album.get("description"):
        console.print(f"[italic]{album['description']}[/italic]")
    if album.get("rationale"):
        console.print(f"\n[dim]Rationale:[/dim] {album['rationale']}")


def print_stats(stats: dict) -> None:
    table = Table(title="Library Statistics", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for key, val in stats.items():
        table.add_row(key.replace("_", " ").title(), str(val))
    console.print(table)


def print_duplicate_groups(groups: list[list[dict]]) -> None:
    if not groups:
        console.print("[green]No near-duplicates found.[/green]")
        return
    console.print(f"[yellow]Found {len(groups)} duplicate group(s):[/yellow]\n")
    for i, group in enumerate(groups, 1):
        console.print(f"[bold]Group {i}[/bold] ({len(group)} photos):")
        for p in group:
            quality = f"{p.get('quality_score', 0.0):.3f}" if p.get("quality_score") is not None else "N/A"
            console.print(f"  [{quality}] {p.get('file_path', '')}")
        console.print()
