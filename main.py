"""
The Librarian — Main Entry Point

CLI-based interface with two operating modes:

1. Full demo (API key set): Chat with Working Agent, Librarian handles memory
2. Middleware mode (no API key): REPL for ingesting messages and searching

Usage:
    python main.py                       # Auto-detect mode
    python main.py --debug               # Debug mode (show Librarian activity)
    python main.py --db my_rolodex.db    # Custom database path
    python main.py --resume <session_id> # Resume a previous session
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.prompt import Prompt

from src.core.orchestrator import Orchestrator
from src.utils.config import LibrarianConfig
from src.utils import logging as lib_log

console = Console()


async def main():
    """Main CLI loop."""
    # Parse args
    debug_mode = "--debug" in sys.argv
    db_path = "rolodex.db"
    resume_id = None
    for i, arg in enumerate(sys.argv):
        if arg == "--db" and i + 1 < len(sys.argv):
            db_path = sys.argv[i + 1]
        if arg == "--resume" and i + 1 < len(sys.argv):
            resume_id = sys.argv[i + 1]

    # Load config (resolve .env relative to this script, not CWD)
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    config = LibrarianConfig.from_env(env_path=env_file)
    config.debug_mode = debug_mode or config.debug_mode
    config.db_path = db_path

    # Validate (now returns warnings, not fatal errors)
    warnings = config.validate()
    for w in warnings:
        console.print(f"[yellow]{w}[/yellow]")

    # Initialize orchestrator
    orchestrator = Orchestrator(config)

    # Determine mode
    middleware_mode = not orchestrator.has_working_agent
    if middleware_mode:
        console.print(
            "\n[bold blue]Running in middleware mode[/bold blue] "
            "[dim](no API key — memory operations only)[/dim]"
        )
        console.print(
            "[dim]Type messages to ingest them. Use /search to query memory. "
            "Set ANTHROPIC_API_KEY for full demo mode.[/dim]\n"
        )

    # Phase 4: handle --resume
    if resume_id:
        full_id = orchestrator.find_session(resume_id) or resume_id
        session_info = orchestrator.resume_session(full_id)
        if session_info:
            lib_log.print_session_resumed(session_info)
        else:
            console.print(f"[yellow]Session '{resume_id}' not found. Starting new session.[/yellow]\n")

    # Print welcome
    lib_log.print_welcome()
    if config.debug_mode:
        console.print("[dim]Debug mode enabled — Librarian activity will be shown.[/dim]\n")

    # Show mode
    mode_str = "full" if orchestrator.has_working_agent else "verbatim"
    console.print(f"[dim]Mode: {mode_str} | Embedding: {config.embedding_strategy}[/dim]\n")

    # Phase 4: show past session count
    past_sessions = orchestrator.list_sessions(limit=100)
    if len(past_sessions) > 1:
        console.print(f"[dim]{len(past_sessions) - 1} past session(s) in rolodex. Use /sessions to browse.[/dim]\n")

    librarian_announced = False

    # Main loop
    try:
        while True:
            # Read user input
            try:
                prompt_label = "[bold cyan]You[/bold cyan]" if not middleware_mode else "[bold blue]Input[/bold blue]"
                user_input = Prompt.ask(prompt_label)
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle special commands
            if user_input.startswith("/"):
                handled = await handle_command(user_input, orchestrator, middleware_mode)
                if handled == "exit":
                    break
                if handled:
                    continue

            # Check if Librarian just activated
            was_active = orchestrator.state.librarian_active
            orchestrator.state.should_activate_librarian(
                config.librarian_activation_tokens
            )
            if orchestrator.state.librarian_active and not was_active and not librarian_announced:
                lib_log.print_librarian_activation()
                librarian_announced = True

            # Process message
            try:
                result = await orchestrator.process_message(user_input)
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                if config.debug_mode:
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")
                continue

            # Show debug info
            if config.debug_mode:
                lib_log.print_debug_info(result["debug"])

            # Show response (or middleware feedback)
            if result["response"] is not None:
                console.print(f"\n[bold green]Assistant[/bold green]: {result['response']}\n")
            else:
                # Middleware mode: show what was indexed
                indexed = result["debug"].get("entries_indexed", 0)
                console.print(f"[dim]Ingested. {indexed} entries extracted. Use /search to query.[/dim]\n")

    finally:
        await orchestrator.shutdown()
        console.print("\n[dim]Session ended. Rolodex saved.[/dim]")


async def handle_command(command: str, orchestrator: Orchestrator, middleware_mode: bool = False) -> str:
    """Handle /commands. Returns 'exit' to quit, truthy if handled."""
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/exit" or cmd == "/quit":
        return "exit"

    elif cmd == "/stats":
        stats = orchestrator.get_stats()
        lib_log.print_stats(stats)
        return "handled"

    elif cmd == "/search":
        if not arg:
            console.print("[yellow]Usage: /search <query>[/yellow]")
            return "handled"
        response = await orchestrator.search_rolodex(arg)
        lib_log.print_search_results(response)
        return "handled"

    elif cmd == "/sessions":
        sessions = orchestrator.list_sessions(limit=20)
        lib_log.print_session_list(sessions, orchestrator.state.conversation_id)
        return "handled"

    elif cmd == "/resume":
        if not arg:
            console.print("[yellow]Usage: /resume <session_id or prefix>[/yellow]")
            return "handled"
        full_id = orchestrator.find_session(arg.strip()) or arg.strip()
        session_info = orchestrator.resume_session(full_id)
        if session_info:
            lib_log.print_session_resumed(session_info)
            console.print(f"[green]Resumed session with {session_info.message_count} messages.[/green]\n")
        else:
            console.print(f"[yellow]Session '{arg}' not found. Use /sessions to list available sessions.[/yellow]")
        return "handled"

    elif cmd == "/ingest" and middleware_mode:
        if not arg:
            console.print("[yellow]Usage: /ingest <role>:<message>  (e.g., /ingest user:Hello world)[/yellow]")
            return "handled"
        if ":" in arg:
            role, content = arg.split(":", 1)
            role = role.strip().lower()
            if role not in ("user", "assistant"):
                console.print("[yellow]Role must be 'user' or 'assistant'[/yellow]")
                return "handled"
            entries = await orchestrator.middleware.ingest(role, content.strip())
            orchestrator.state = orchestrator.middleware.state
            console.print(f"[dim]Ingested as {role}. {len(entries)} entries extracted.[/dim]")
        else:
            console.print("[yellow]Usage: /ingest <role>:<message>[/yellow]")
        return "handled"

    elif cmd == "/debug":
        orchestrator.config.debug_mode = not orchestrator.config.debug_mode
        state = "enabled" if orchestrator.config.debug_mode else "disabled"
        console.print(f"[blue]Debug mode {state}[/blue]")
        return "handled"

    elif cmd == "/help":
        help_text = (
            "\n[bold]Commands:[/bold]\n"
            "  /stats          Show session statistics\n"
            "  /search <query> Search the rolodex directly\n"
            "  /sessions       List past sessions\n"
            "  /resume <id>    Resume a previous session\n"
            "  /debug          Toggle debug output\n"
            "  /help           Show this help\n"
            "  /exit           End session\n"
        )
        if middleware_mode:
            help_text += "  /ingest r:msg   Ingest a message (role:content)\n"
        console.print(help_text)
        return "handled"

    else:
        console.print(f"[yellow]Unknown command: {cmd}. Try /help[/yellow]")
        return "handled"


if __name__ == "__main__":
    asyncio.run(main())
