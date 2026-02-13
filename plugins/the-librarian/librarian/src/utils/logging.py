


from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from ..core.types import RolodexEntry, LibrarianResponse, SessionInfo
console = Console()
def print_debug_info(debug: Dict[str, Any]):

    if not any([
        debug.get("entries_indexed", 0) > 0,
        debug.get("gap_detected", False),
        debug.get("retrieval_performed", False),
    ]):

        if debug.get("librarian_active"):
            console.print("[dim]  📚 Librarian: monitoring (no action needed)[/dim]")
        return
    lines = []

    n_indexed = debug.get("entries_indexed", 0)
    if n_indexed > 0:
        lines.append(f"[green]📥 Indexed {n_indexed} new entries[/green]")

    if debug.get("gap_detected"):
        topic = debug.get("gap_topic", "unknown")
        lines.append(f"[yellow]🔍 Gap detected: \"{topic}\"[/yellow]")

    if debug.get("retrieval_performed"):
        if debug.get("retrieval_found"):
            n_results = debug.get("retrieval_entries", 0)
            time_ms = debug.get("search_time_ms", 0)
            cache = "cache hit" if debug.get("cache_hit") else "cold storage"
            lines.append(
                f"[green]✅ Retrieved {n_results} entries ({cache}, {time_ms:.0f}ms)[/green]"
            )
        else:
            lines.append("[red]❌ No matching entries found — falling back to user[/red]")

    if debug.get("preload_performed"):
        strategy = debug.get("preload_strategy", "none")
        pressure = debug.get("preload_pressure", 0)
        injected = debug.get("preload_injected", 0)
        cached = debug.get("preload_cached", 0)
        pressure_bar = "▓" * int(pressure * 10) + "░" * (10 - int(pressure * 10))
        lines.append(
            f"[cyan]🔮 Preload: {strategy} strategy "
            f"[{pressure_bar}] {pressure:.0%} pressure[/cyan]"
        )
        if injected > 0:
            lines.append(
                f"[cyan]   ↳ Injected {injected} high-confidence entries proactively[/cyan]"
            )
        if cached > 0:
            lines.append(
                f"[cyan]   ↳ Warmed {cached} entries into hot cache[/cyan]"
            )

    tier_events = debug.get("tier_events", [])
    if tier_events:
        for event in tier_events:
            direction = "⬆️ Promoted" if event.new_tier.value == "hot" else "⬇️ Demoted"
            lines.append(
                f"[magenta]{direction}: entry {event.entry_id[:8]}… "
                f"(score: {event.score:.2f})[/magenta]"
            )

    if debug.get("tier_sweep_performed"):
        summary = debug.get("tier_sweep_summary", {})
        lines.append(
            f"[blue]🔄 Tier sweep: scanned {summary.get('scanned', 0)}, "
            f"promoted {summary.get('promoted', 0)}, "
            f"demoted {summary.get('demoted', 0)}[/blue]"
        )

    cross_session = debug.get("cross_session_results", 0)
    if cross_session > 0:
        lines.append(
            f"[blue]🌐 {cross_session} result(s) from prior sessions[/blue]"
        )

    if debug.get("indexing_error"):
        lines.append(f"[red]⚠ Indexing error: {debug['indexing_error']}[/red]")
    content = "\n".join(lines)
    console.print(Panel(content, title="[bold blue]📚 Librarian[/bold blue]",
                        border_style="blue", padding=(0, 1)))
def print_stats(stats: Dict[str, Any]):

    table = Table(title="📊 The Librarian — Session Stats", border_style="blue")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Conversation ID", str(stats.get("conversation_id", ""))[:8] + "...")
    table.add_row("Total Messages", str(stats.get("total_messages", 0)))
    table.add_row("Total Tokens", f"{stats.get('total_tokens', 0):,}")
    table.add_row("Librarian Active", "✅" if stats.get("librarian_active") else "❌")
    table.add_row("Rolodex Entries", str(stats.get("total_entries", 0)))
    table.add_row("Hot Cache Entries", str(stats.get("hot_cache_entries", 0)))
    table.add_row("Cold Storage Entries", str(stats.get("cold_storage_entries", 0)))
    table.add_row("Entries Created (session)", str(stats.get("total_entries_created", 0)))
    table.add_row("Last Indexed Turn", str(stats.get("last_indexed_turn", 0)))

    tier_dist = stats.get("tier_distribution", {})
    if tier_dist:
        tier_str = ", ".join(f"{k}: {v}" for k, v in tier_dist.items())
        table.add_row("Tier Distribution", tier_str)
    avg_score = stats.get("avg_hot_score", 0)
    if avg_score:
        table.add_row("Avg HOT Score", f"{avg_score:.3f}")

    pressure_info = stats.get("pressure", {})
    if pressure_info:
        table.add_row("Session Pressure", f"{pressure_info.get('pressure', 0):.1%}")
        table.add_row("Preload Strategy", pressure_info.get("strategy", "none"))
        table.add_row("Total Gaps", str(pressure_info.get("total_gaps", 0)))
        hit_rate = pressure_info.get("cache_hit_rate", 0)
        table.add_row("Cache Hit Rate", f"{hit_rate:.0%}")

    total_sessions = stats.get("total_sessions", 0)
    if total_sessions:
        table.add_row("Total Sessions", str(total_sessions))
    cross_search = stats.get("cross_session_search", False)
    table.add_row("Cross-Session Search", "✅" if cross_search else "❌")
    categories = stats.get("categories", {})
    if categories:
        cats_str = ", ".join(f"{k}: {v}" for k, v in categories.items())
        table.add_row("Categories", cats_str)
    console.print(table)
def print_search_results(response: LibrarianResponse):

    if not response.found:
        console.print("[yellow]No results found.[/yellow]")
        return
    console.print(f"\n[bold]Found {len(response.entries)} entries[/bold] "
                  f"({response.search_time_ms:.0f}ms, "
                  f"{'cache hit' if response.cache_hit else 'cold storage'})\n")
    for i, entry in enumerate(response.entries, 1):
        category = entry.category.value.upper()
        tags = ", ".join(entry.tags) if entry.tags else ""
        panel_title = f"[{i}] [{category}]"
        if tags:
            panel_title += f"  Tags: {tags}"
        console.print(Panel(
            entry.content,
            title=panel_title,
            border_style="dim",
            padding=(0, 1),
        ))
def print_welcome():

    console.print()
    console.print(Panel(
        "[bold]The Librarian[/bold] — Three-Tier Memory Architecture\n"
        "[dim]Your conversation is being indexed for perfect recall.[/dim]\n\n"
        "Commands: /stats  /search <query>  /sessions  /resume <id>  /debug  /exit",
        border_style="blue",
        padding=(1, 2),
    ))
    console.print()


def print_librarian_activation():

    console.print(Panel(
        "[bold green]📚 Librarian is now active[/bold green]\n"
        "[dim]Indexing conversation content. Memory retrieval enabled.[/dim]",
        border_style="green",
        padding=(0, 1),
    ))


def print_session_list(
    sessions: List[SessionInfo],
    current_session_id: Optional[str] = None,
):

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(
        title="📚 Past Sessions",
        border_style="blue",
    )
    table.add_column("ID", style="bold", width=10)
    table.add_column("Status", width=8)
    table.add_column("Messages", justify="right", width=10)
    table.add_column("Entries", justify="right", width=8)
    table.add_column("Last Active", width=20)
    table.add_column("Summary", max_width=40)

    for session in sessions:
        short_id = session.session_id[:8]
        is_current = session.session_id == current_session_id

        status = "[green]active[/green]" if session.status == "active" else "[dim]ended[/dim]"
        if is_current:
            status = "[bold green]current[/bold green]"
            short_id = f"[bold]{short_id}[/bold]"

        last_active = ""
        if session.last_active:
            last_active = session.last_active.strftime("%Y-%m-%d %H:%M")

        summary = session.summary[:40] if session.summary else "[dim]—[/dim]"

        table.add_row(
            short_id,
            status,
            str(session.message_count),
            str(session.entry_count),
            last_active,
            summary,
        )

    console.print(table)
    console.print("[dim]Use /resume <id> to resume a session.[/dim]\n")


def print_session_resumed(session_info: SessionInfo):

    short_id = session_info.session_id[:8]
    started = session_info.started_at.strftime("%Y-%m-%d %H:%M")
    console.print(Panel(
        f"[bold green]📂 Resumed session {short_id}...[/bold green]\n"
        f"[dim]Started: {started} | "
        f"Messages: {session_info.message_count} | "
        f"Entries: {session_info.entry_count}[/dim]",
        border_style="green",
        padding=(0, 1),
    ))
