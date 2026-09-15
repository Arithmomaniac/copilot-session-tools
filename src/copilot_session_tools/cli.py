"""Command-line interface for Copilot Session Tools.

This module provides a modern CLI built with Typer for scanning, searching,
and exporting VS Code GitHub Copilot chat history.
"""

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from copilot_session_tools import (
    ChatMessage,
    ChatSession,
    Database,
    __version__,
    export_session_to_file,
    export_session_to_html_file,
    generate_session_filename,
    generate_session_html_filename,
    get_vscode_storage_paths,
)
from copilot_session_tools.content_types import (
    CONTENT_TYPES,
    SEARCH_CONTENT_TYPES,
    resolve_content_set,
    resolve_search_content_set,
)
from copilot_session_tools.refresh import (
    DEFAULT_PARSE_WORKERS,
    enrich_single_session,
    parse_single_cli_session,
    run_enrichment,
    run_refresh,
)
from copilot_session_tools.sources import DEFAULT_SOURCE_ID, discover_source_presets

# On Windows, reconfigure stdout/stderr to UTF-8 when piped to prevent
# Rich from falling back to cp1252 which can't handle Unicode output
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[call-non-callable]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[call-non-callable]


from copilot_session_tools.utils import DEFAULT_DB_PATH as _DEFAULT_DB_PATH
from copilot_session_tools.utils import format_timestamp


def _default_db_path() -> Path:
    """Return the default database path: ~/.copilot/copilot-session-tools.db"""
    return _DEFAULT_DB_PATH


_DEFAULT_DB = _DEFAULT_DB_PATH


def _ensure_db_exists(db: Path) -> None:
    """Check that the database file exists, with a friendly error if not."""
    try:
        exists = db.exists()
    except FileNotFoundError:
        exists = False
    if not exists:
        typer.echo(f"Error: Database not found at {db}", err=True)
        typer.echo(
            "Run 'copilot-session-tools scan' first to create the database.",
            err=True,
        )
        raise typer.Exit(code=2)


def _make_database_for_command(db: Path, *, allow_create: bool = False) -> "Database":
    """Create a Database for a command, optionally bootstrapping the DB file."""
    if allow_create:
        db.parent.mkdir(parents=True, exist_ok=True)
    else:
        _ensure_db_exists(db)
    return _make_database(db)


app = typer.Typer(
    name="copilot-session-tools",
    help="Create a searchable archive of VS Code GitHub Copilot chats.",
    no_args_is_help=True,
)
sources_app = typer.Typer(help="Manage named Copilot CLI application sources.")
app.add_typer(sources_app, name="sources")
console = Console()

# Module-level state set by the app callback
_unenriched_only: bool = False
_chronicle_db: Path | None = None


def _make_database(db: str | Path) -> "Database":
    """Create a Database with the current global settings."""
    return Database(db, unenriched_only=_unenriched_only, chronicle_db_path=_chronicle_db)


def _validate_rescan_session(
    rescan_session_id: str | None,
    *,
    expected_session_id: str | None = None,
) -> ChatSession | None:
    """Validate and parse a targeted CLI session before DB creation."""
    if rescan_session_id is None:
        return None

    if expected_session_id is not None and rescan_session_id != expected_session_id:
        typer.echo(
            f"Error: --rescan-session ({rescan_session_id}) must match --session-id ({expected_session_id}).",
            err=True,
        )
        raise typer.Exit(1)

    parsed = parse_single_cli_session(rescan_session_id)
    if isinstance(parsed, str):
        typer.echo(f"Error: {parsed}", err=True)
        raise typer.Exit(1)

    return parsed


def _prepare_database_for_command(
    db: Path,
    *,
    rescan_session_id: str | None = None,
    expected_session_id: str | None = None,
) -> "Database":
    """Create a Database and apply a targeted rescan when requested."""
    parsed_rescan = _validate_rescan_session(rescan_session_id, expected_session_id=expected_session_id)
    database = _make_database_for_command(db, allow_create=parsed_rescan is not None)
    if parsed_rescan is not None:
        database.enrich_session(parsed_rescan)
    return database


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"copilot-session-tools version {__version__}")
        raise typer.Exit()


def _print_upgrade_notice():
    """Print a notice if a newer version is available on PyPI."""
    try:
        from copilot_session_tools.version_check import check_for_upgrade

        latest = check_for_upgrade()
        if latest:
            console.print(f"\n[yellow]💡 A newer version (v{latest}) is available. Upgrade with: pip install --upgrade copilot-session-tools[/yellow]")
    except Exception:  # noqa: S110
        pass


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
    unenriched_only: Annotated[
        bool,
        typer.Option(
            "--unenriched-only",
            help="Disable cst_* table reads; use Chronicle tables only.",
        ),
    ] = False,
    chronicle_db: Annotated[
        Path | None,
        typer.Option(
            "--chronicle-db",
            help="Path to Copilot CLI Chronicle session-store.db (auto-detected by default).",
        ),
    ] = None,
):
    """Copilot Session Tools - Create a searchable archive of VS Code GitHub Copilot chats."""
    global _unenriched_only, _chronicle_db  # noqa: PLW0603
    _unenriched_only = unenriched_only
    _chronicle_db = chronicle_db


@app.command()
def scan(
    db: Annotated[
        Path,
        typer.Option(
            "--db",
            "-d",
            help="Path to SQLite database file.",
        ),
    ] = _DEFAULT_DB,
    storage_path: Annotated[
        list[Path] | None,
        typer.Option(
            "--storage-path",
            "-s",
            help="Custom VS Code storage path(s) to scan. Can be specified multiple times.",
            exists=True,
            file_okay=False,
        ),
    ] = None,
    edition: Annotated[
        str,
        typer.Option(
            "--edition",
            "-e",
            help="VS Code edition to scan.",
        ),
    ] = "both",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show verbose output.",
        ),
    ] = False,
    full: Annotated[
        bool,
        typer.Option(
            "--full",
            "-f",
            help="Full scan: update all sessions regardless of file changes.",
        ),
    ] = False,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-w",
            help="Number of parallel worker processes for parsing. Default: min(4, cores/2).",
            min=1,
        ),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option(
            "--source",
            help="Scan one named Copilot CLI source and skip VS Code scanning.",
        ),
    ] = None,
):
    """Scan for and import Copilot chat sessions, enriching them into the built-in store.

    By default, uses incremental refresh: only updates sessions whose source files
    have changed (based on file mtime and size). Use --full to force a complete
    re-import of all sessions.

    \b
    Examples:
      copilot-session-tools scan
      copilot-session-tools scan --edition stable
      copilot-session-tools scan --full --verbose
      copilot-session-tools scan --storage-path /custom/path
      copilot-session-tools scan --workers 8
    """
    if edition not in ("stable", "insider", "both"):
        console.print("[red]Error: edition must be 'stable', 'insider', or 'both'[/red]")
        console.print("[dim]  copilot-session-tools scan --edition stable[/dim]")
        raise typer.Exit(1)

    db.parent.mkdir(parents=True, exist_ok=True)
    database = _make_database(db)

    selected_source = None
    if source:
        if storage_path or edition != "both":
            console.print("[red]Error: --source cannot be combined with --storage-path or --edition.[/red]")
            raise typer.Exit(1)
        selected_source = database.get_source(source)
        if selected_source is None:
            console.print(f"[red]Error: Source not found: {source}[/red]")
            raise typer.Exit(1)
        if not selected_source.enabled:
            console.print(f"[red]Error: Source is disabled: {selected_source.name}[/red]")
            raise typer.Exit(1)
        paths = []
    elif storage_path:
        paths = [(str(p), "custom") for p in storage_path]
    else:
        all_paths = get_vscode_storage_paths()
        if edition == "both":
            paths = all_paths
        else:
            paths = [(p, e) for p, e in all_paths if e == edition]

    # Resolve worker count
    n_workers = workers if workers is not None else DEFAULT_PARSE_WORKERS

    console.print("Scanning for Copilot chat sessions...")
    if full:
        console.print("  (Full mode: will update all sessions)")
    else:
        console.print("  (Incremental mode: skipping unchanged sessions)")
    console.print(f"  Using {n_workers} worker process(es)")
    if verbose:
        for path, ed in paths:
            console.print(f"  Checking: {path} ({ed})")

    def _verbose_progress(event: str, item: object) -> None:
        from copilot_session_tools.scanner.models import ChatSession as _CS
        from copilot_session_tools.scanner.models import SessionFileInfo as _SFI

        if isinstance(item, _CS):
            workspace = item.workspace_name or "Unknown workspace"
            console.print(f"  {event.capitalize()}: {workspace} ({len(item.messages)} messages)")
        elif isinstance(item, _SFI):
            workspace = item.workspace_name or "Unknown workspace"
            console.print(f"  Skipped (unchanged): {workspace}")
        elif event == "enriched":
            console.print(f"  Enriched: {item}")
        elif event == "reparsed":
            console.print(f"  Reparsed: {item}")
        elif event == "enrich_failed":
            console.print(f"  [yellow]{item}[/yellow]")

    progress_cb = _verbose_progress if verbose else None

    result = run_refresh(database, paths, full=full, on_progress=progress_cb, workers=n_workers)
    added = result.added
    updated = result.updated
    skipped = result.skipped

    console.print("\n[green]VS Code import complete:[/green]")
    console.print(f"  Added: {added} sessions")
    console.print(f"  Updated: {updated} sessions")
    console.print(f"  Skipped (unchanged): {skipped} sessions")

    # --- CLI session enrichment from Chronicle's built-in session store ---
    console.print("\n[cyan]Enriching CLI sessions...[/cyan]")
    enrich_result = run_enrichment(
        database,
        on_progress=progress_cb,
        workers=n_workers,
        sources=[selected_source] if selected_source else None,
    )

    console.print(f"  Enriched: {enrich_result.enriched} sessions")
    if enrich_result.reparsed:
        console.print(f"  Reparsed: {enrich_result.reparsed} sessions (parser version upgrade)")
    if enrich_result.failed:
        console.print(f"  Skipped (no events file or parse error): {enrich_result.failed} sessions")
    if enrich_result.orphaned:
        console.print(f"\n[yellow]Cleaned up {enrich_result.orphaned} orphaned session(s)[/yellow]")

    stats = database.get_stats()
    console.print("\n[cyan]Database now contains:[/cyan]")
    console.print(f"  {stats['session_count']} sessions")
    console.print(f"  {stats['message_count']} messages")
    console.print(f"  {stats['workspace_count']} workspaces")

    _print_upgrade_notice()


@app.command()
def enrich(
    session_id: str = typer.Argument(..., help="Session ID to enrich"),
    db: Annotated[
        Path,
        typer.Option(
            "--db",
            "-d",
            help="Path to SQLite database file.",
        ),
    ] = _DEFAULT_DB,
    source: Annotated[
        str | None,
        typer.Option("--source", help="Named Copilot CLI source for a native session ID."),
    ] = None,
):
    """Enrich a single CLI session from its events.jsonl file.

    \b
    Examples:
      copilot-session-tools enrich a1b2c3d4-e5f6-7890-abcd-ef1234567890
    """
    _ensure_db_exists(db)
    database = _make_database(db)
    selected_source = database.get_source(source) if source else database.get_source(DEFAULT_SOURCE_ID)
    if source and selected_source is None:
        console.print(f"[red]Error: Source not found: {source}[/red]")
        raise typer.Exit(1)

    error = enrich_single_session(database, session_id, source=selected_source)
    if error:
        console.print(f"[red]Error: {error}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Successfully enriched session {session_id}[/green]")


@sources_app.command("list")
def list_sources(
    db: Annotated[Path, typer.Option("--db", "-d", help="Path to SQLite database file.")] = _DEFAULT_DB,
):
    """List named Copilot CLI sources for an archive."""
    database = _make_database_for_command(db, allow_create=True)
    for source in database.list_sources():
        state = "enabled" if source.enabled else "disabled"
        console.print(f"{source.name}\t{state}\t{source.base_dir}")


@sources_app.command("presets")
def list_source_presets():
    """List application presets detected on this machine."""
    presets = discover_source_presets()
    if not presets:
        console.print("No source presets detected.")
        return
    for preset in presets:
        console.print(f"{preset.preset_id}\t{preset.name}\t{preset.base_dir}")


@sources_app.command("add")
def add_source(
    name: Annotated[str, typer.Option("--name", help="Unique case-insensitive identifier.")],
    base_dir: Annotated[Path | None, typer.Option("--base-dir", help="Application Copilot base directory.")] = None,
    preset: Annotated[str | None, typer.Option("--preset", help="Detected preset ID, such as scout.")] = None,
    db: Annotated[Path, typer.Option("--db", "-d", help="Path to SQLite database file.")] = _DEFAULT_DB,
):
    """Register a named Copilot CLI source."""
    if (base_dir is None) == (preset is None):
        console.print("[red]Error: specify exactly one of --base-dir or --preset.[/red]")
        raise typer.Exit(1)
    if preset:
        detected = {item.preset_id: item for item in discover_source_presets()}
        selected = detected.get(preset.casefold())
        if selected is None:
            console.print(f"[red]Error: Source preset is not available: {preset}[/red]")
            raise typer.Exit(1)
        base_dir = selected.base_dir
    database = _make_database_for_command(db, allow_create=True)
    try:
        if base_dir is None:
            raise RuntimeError("Source base directory was not resolved.")
        source = database.add_source(name, base_dir)
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Added source {source.name}: {source.base_dir}[/green]")


@sources_app.command("rename")
def rename_source(
    source: Annotated[str, typer.Argument(help="Existing source name or ID.")],
    name: Annotated[str, typer.Option("--name", help="New unique case-insensitive identifier.")],
    db: Annotated[Path, typer.Option("--db", "-d", help="Path to SQLite database file.")] = _DEFAULT_DB,
):
    """Rename a source while preserving session identities."""
    database = _make_database_for_command(db)
    try:
        renamed = database.rename_source(source, name)
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Renamed source to {renamed.name}[/green]")


def _set_source_state(source: str, db: Path, *, enabled: bool) -> None:
    database = _make_database_for_command(db)
    try:
        updated = database.set_source_enabled(source, enabled=enabled)
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]{'Enabled' if enabled else 'Disabled'} source {updated.name}[/green]")


@sources_app.command("disable")
def disable_source(
    source: Annotated[str, typer.Argument(help="Source name or ID.")],
    db: Annotated[Path, typer.Option("--db", "-d", help="Path to SQLite database file.")] = _DEFAULT_DB,
):
    """Disable live reads and refresh for a source."""
    _set_source_state(source, db, enabled=False)


@sources_app.command("enable")
def enable_source(
    source: Annotated[str, typer.Argument(help="Source name or ID.")],
    db: Annotated[Path, typer.Option("--db", "-d", help="Path to SQLite database file.")] = _DEFAULT_DB,
):
    """Re-enable live reads and refresh for a source."""
    _set_source_state(source, db, enabled=True)


_CONTENT_TYPES_HELP = "Available types: " + ", ".join(sorted(CONTENT_TYPES))
_SEARCH_CONTENT_TYPES_HELP = "Available types: " + ", ".join(sorted(SEARCH_CONTENT_TYPES))


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    db: Annotated[
        Path,
        typer.Option(
            "--db",
            "-d",
            help="Path to SQLite database file.",
        ),
    ] = _DEFAULT_DB,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-l",
            "--top",
            help="Maximum number of results to show (top).",
        ),
    ] = 20,
    skip: Annotated[
        int,
        typer.Option(
            "--skip",
            help="Number of results to skip (for pagination).",
        ),
    ] = 0,
    role: Annotated[
        str | None,
        typer.Option(
            "--role",
            "-r",
            help="Filter by message role (user or assistant).",
        ),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option("--source", help="Filter results to one named Copilot CLI source."),
    ] = None,
    title_filter: Annotated[
        str | None,
        typer.Option(
            "--title",
            "-t",
            help="Filter by session title or workspace name.",
        ),
    ] = None,
    repository_filter: Annotated[
        str | None,
        typer.Option(
            "--repository",
            "--repo",
            help="Filter by repository URL (e.g., github.com/owner/repo).",
        ),
    ] = None,
    include: Annotated[
        list[str] | None,
        typer.Option(
            "--include",
            "-i",
            help=f"Content types to search in (exclusive — replaces defaults). Comma-separated or repeated. {_SEARCH_CONTENT_TYPES_HELP}",
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude",
            "-x",
            help=f"Content types to exclude from search. Comma-separated or repeated. {_SEARCH_CONTENT_TYPES_HELP}",
        ),
    ] = None,
    full_content: Annotated[
        bool,
        typer.Option(
            "--full",
            "-F",
            help="Show full content instead of truncated snippets.",
        ),
    ] = False,
    sort_by: Annotated[
        str,
        typer.Option(
            "--sort",
            "-s",
            help="Sort results by relevance (default) or date.",
        ),
    ] = "relevance",
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output results as JSON for programmatic consumption.",
        ),
    ] = False,
    rescan_session: Annotated[
        str | None,
        typer.Option(
            "--rescan-session",
            help="Parse and enrich one CLI session by ID before searching.",
        ),
    ] = None,
):
    """Search chat messages in the database.

    Supports advanced query syntax:

    \b
    - Multiple words: "python function" matches both words (AND logic)
    - Exact phrases: Use quotes like "python function" for exact match
    - Field filters in query: role:user, role:assistant, workspace:name, title:name, repository:url (or repo:url)
    - Date filters: start_date:2024-01-01 end_date:2024-12-31 (yyyy-mm-dd format, inclusive)

    Examples:

    \b
      copilot-session-tools search "python function"
      copilot-session-tools search "role:user python"
      copilot-session-tools search "workspace:my-project"
      copilot-session-tools search "repo:github.com/owner/repo"
      copilot-session-tools search "start_date:2024-01-01 end_date:2024-06-30"
      copilot-session-tools search '"exact phrase"'
      copilot-session-tools search "python function" --json
      copilot-session-tools search "tool result" --rescan-session a1b2c3d4-...

    Use --role to filter by user requests or assistant responses.
    Use --title to filter by session/workspace name.
    Use --repository to filter by git repository URL.
    Use --skip and --limit/--top for pagination.
    Use --include / --exclude to control which content types are searched.
    Use --full to show complete content instead of truncated snippets.
    Use --sort to sort by relevance (default) or date.
    Use --json to output results as JSON for programmatic consumption.
    Use --rescan-session to refresh one CLI session before searching.
    """
    if role and role not in ("user", "assistant"):
        console.print("[red]Error: role must be 'user' or 'assistant'[/red]")
        console.print('[dim]  copilot-session-tools search "query" --role user[/dim]')
        raise typer.Exit(1)

    if sort_by not in ("relevance", "date"):
        console.print("[red]Error: sort must be 'relevance' or 'date'[/red]")
        console.print('[dim]  copilot-session-tools search "query" --sort relevance[/dim]')
        raise typer.Exit(1)

    search_content_set = resolve_search_content_set(include, exclude)

    database = _prepare_database_for_command(db, rescan_session_id=rescan_session)
    if source and database.get_source(source) is None:
        console.print(f"[red]Error: Source not found: {source}[/red]")
        raise typer.Exit(1)
    results = database.search(
        query,
        limit=limit,
        skip=skip,
        role=role,
        search_content_set=search_content_set,
        session_title=title_filter,
        sort_by=sort_by,
        repository=repository_filter,
        source_name=source,
    )

    if json_output:
        import json

        print(json.dumps(results, ensure_ascii=False, default=str))
        return

    if not results:
        console.print(f"[yellow]No results found for '{query}'[/yellow]")
        return

    # Display result count with pagination info
    start_num = skip + 1
    end_num = skip + len(results)
    console.print(f"[green bold]Showing results {start_num}-{end_num} for '{query}':[/green bold]\n")

    for i, result in enumerate(results, start_num):
        console.print(f"[cyan bold]━━━ Result {i} ━━━[/cyan bold]")
        console.print(f"[bright_blue bold]Session ID:[/bright_blue bold] {result['session_id']}")

        if result.get("workspace_name"):
            console.print(f"[bright_blue bold]Workspace:[/bright_blue bold]  [yellow]{result['workspace_name']}[/yellow]")

        if result.get("custom_title"):
            console.print(f"[bright_blue bold]Title:[/bright_blue bold]      {result['custom_title']}")

        if result.get("created_at"):
            formatted_date = format_timestamp(result["created_at"])
            console.print(f"[bright_blue bold]Date:[/bright_blue bold]       [dim]{formatted_date}[/dim]")

        role_color = "green" if result["role"] == "user" else "magenta"
        console.print(f"[bright_blue bold]Role:[/bright_blue bold]       [{role_color}]{result['role']}[/{role_color}]")

        if result.get("match_type") and result["match_type"] != "message":
            console.print(f"[bright_blue bold]Match Type:[/bright_blue bold] [cyan]{result['match_type']}[/cyan]")

        content = result["content"]
        if not full_content and len(content) > 200:
            content = content[:200] + "[dim]... (use --full to see more)[/dim]"
        console.print(f"[bright_blue bold]Content:[/bright_blue bold]    {content}")
        console.print()


@app.command()
def stats(
    db: Annotated[
        Path,
        typer.Option(
            "--db",
            "-d",
            help="Path to SQLite database file.",
        ),
    ] = _DEFAULT_DB,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output statistics as JSON for programmatic consumption.",
        ),
    ] = False,
):
    """Show database statistics.

    \b
    Examples:
      copilot-session-tools stats
      copilot-session-tools stats --json
      copilot-session-tools stats --json | jq '.workspaces'
    """
    _ensure_db_exists(db)
    database = _make_database(db)
    stats_data = database.get_stats()

    if json_output:
        import json

        workspaces = database.get_workspaces()
        repositories = database.get_repositories()
        output = {
            **stats_data,
            "workspaces": workspaces,
            "repositories": repositories,
        }
        print(json.dumps(output, ensure_ascii=False, default=str))
        return

    console.print("[bold]Database Statistics:[/bold]")
    console.print(f"  Sessions: {stats_data['session_count']}")
    console.print(f"  Messages: {stats_data['message_count']}")
    console.print(f"  Workspaces: {stats_data['workspace_count']}")

    if stats_data["editions"]:
        console.print("\n  [cyan]By VS Code Edition:[/cyan]")
        for edition, count in stats_data["editions"].items():
            console.print(f"    {edition}: {count}")

    workspaces = database.get_workspaces()
    if workspaces:
        console.print("\n  [cyan]Workspaces:[/cyan]")
        for ws in workspaces[:10]:
            console.print(f"    {ws['workspace_name']}: {ws['session_count']} sessions")
        if len(workspaces) > 10:
            console.print(f"    ... and {len(workspaces) - 10} more")

    repositories = database.get_repositories()
    if repositories:
        console.print("\n  [green]Repositories:[/green]")
        for repo in repositories[:10]:
            console.print(f"    {repo['repository_url']}: {repo['session_count']} sessions")
        if len(repositories) > 10:
            console.print(f"    ... and {len(repositories) - 10} more")


@app.command()
def export(
    db: Annotated[
        Path,
        typer.Option(
            "--db",
            "-d",
            help="Path to SQLite database file.",
        ),
    ] = _DEFAULT_DB,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Output file (- for stdout).",
        ),
    ] = "-",
):
    """Export the database as JSON.

    \b
    Examples:
      copilot-session-tools export
      copilot-session-tools export -o sessions.json
      copilot-session-tools export -o - | jq '.[] | .session_id'
    """
    _ensure_db_exists(db)
    database = _make_database(db)
    json_data = database.export_json()

    if output == "-":
        console.print(json_data)
    else:
        Path(output).write_text(json_data, encoding="utf-8")
        console.print(f"[green]Exported to {output}[/green]")


@app.command("export-markdown")
def export_markdown(
    db: Annotated[
        Path,
        typer.Option(
            "--db",
            "-d",
            help="Path to SQLite database file.",
        ),
    ] = _DEFAULT_DB,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for markdown files.",
        ),
    ] = Path(),
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            "-s",
            help="Export only a specific session by ID.",
        ),
    ] = None,
    rescan_session: Annotated[
        str | None,
        typer.Option(
            "--rescan-session",
            help="Parse and enrich one CLI session by ID before exporting.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show verbose output.",
        ),
    ] = False,
    include: Annotated[
        list[str] | None,
        typer.Option(
            "--include",
            "-i",
            help=f"Content types to include (comma-separated or repeated). {_CONTENT_TYPES_HELP}",
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude",
            "-x",
            help=f"Content types to exclude (comma-separated or repeated). {_CONTENT_TYPES_HELP}",
        ),
    ] = None,
):
    """Export sessions as markdown files.

    Each session is exported to a separate markdown file with:
    - Header block with metadata (session ID, workspace, dates)
    - Messages separated by horizontal rules
    - Message numbers and roles as bold headers
    - Tool call summaries in italics
    - Thinking block notices in italics (content omitted)

    Use --include / --exclude to control which content types appear.
    Default includes: agent-details.

    \b
    Examples:
      copilot-session-tools export-markdown -o ./exports
      copilot-session-tools export-markdown --session-id a1b2c3d4-... -o .
      copilot-session-tools export-markdown --session-id a1b2c3d4-... --rescan-session a1b2c3d4-... -o .
      copilot-session-tools export-markdown --exclude tool-results,thinking
    """
    database = _prepare_database_for_command(db, rescan_session_id=rescan_session, expected_session_id=session_id)

    content_set = resolve_content_set(include, exclude)

    output_dir.mkdir(parents=True, exist_ok=True)

    if session_id:
        session = database.get_session(session_id)
        if session is None:
            console.print(f"[red]Error: Session '{session_id}' not found.[/red]")
            console.print("[dim]  List sessions: copilot-session-tools stats[/dim]")
            raise typer.Exit(1)

        filename = generate_session_filename(session)
        file_path = output_dir / filename
        export_session_to_file(session, file_path, content_set=content_set)
        console.print(f"[green]Exported: {file_path}[/green]")
    else:
        sessions = database.list_sessions()
        exported = 0

        for session_info in sessions:
            session = database.get_session(session_info["session_id"])
            if session:
                filename = generate_session_filename(session)
                file_path = output_dir / filename
                export_session_to_file(session, file_path, content_set=content_set)
                exported += 1
                if verbose:
                    console.print(f"  Exported: {file_path}")

        console.print(f"\n[green]Exported {exported} sessions to {output_dir}/[/green]")


@app.command("export-html")
def export_html(
    db: Annotated[
        Path,
        typer.Option(
            "--db",
            "-d",
            help="Path to SQLite database file.",
        ),
    ] = _DEFAULT_DB,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for HTML files.",
        ),
    ] = Path(),
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            "-s",
            help="Export only a specific session by ID.",
        ),
    ] = None,
    rescan_session: Annotated[
        str | None,
        typer.Option(
            "--rescan-session",
            help="Parse and enrich one CLI session by ID before exporting.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show verbose output.",
        ),
    ] = False,
    include: Annotated[
        list[str] | None,
        typer.Option(
            "--include",
            "-i",
            help=f"Content types to include (comma-separated or repeated). {_CONTENT_TYPES_HELP}",
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude",
            "-x",
            help=f"Content types to exclude (comma-separated or repeated). {_CONTENT_TYPES_HELP}",
        ),
    ] = None,
):
    """Export sessions as self-contained static HTML files.

    Each session is exported to a separate HTML file with the same rich
    rendering as the web viewer, but without interactive elements (toolbar,
    copy buttons, AJAX). The HTML is self-contained with no external
    dependencies.

    Use --include / --exclude to control which content types appear.
    Default includes: agent-details.

    \b
    Examples:
      copilot-session-tools export-html -o ./exports
      copilot-session-tools export-html --session-id a1b2c3d4-... -o .
      copilot-session-tools export-html --session-id a1b2c3d4-... --rescan-session a1b2c3d4-... -o .
      copilot-session-tools export-html --exclude tool-results,thinking
    """
    database = _prepare_database_for_command(db, rescan_session_id=rescan_session, expected_session_id=session_id)

    content_set = resolve_content_set(include, exclude)

    output_dir.mkdir(parents=True, exist_ok=True)

    if session_id:
        session = database.get_session(session_id)
        if session is None:
            console.print(f"[red]Error: Session '{session_id}' not found.[/red]")
            console.print("[dim]  List sessions: copilot-session-tools stats[/dim]")
            raise typer.Exit(1)

        filename = generate_session_html_filename(session)
        file_path = output_dir / filename
        export_session_to_html_file(session, file_path, content_set=content_set)
        console.print(f"[green]Exported: {file_path}[/green]")
    else:
        sessions = database.list_sessions()
        exported = 0

        for session_info in sessions:
            session = database.get_session(session_info["session_id"])
            if session:
                filename = generate_session_html_filename(session)
                file_path = output_dir / filename
                export_session_to_html_file(session, file_path, content_set=content_set)
                exported += 1
                if verbose:
                    console.print(f"  Exported: {file_path}")

        console.print(f"\n[green]Exported {exported} sessions to {output_dir}/[/green]")


@app.command("import-json")
def import_json(
    json_file: Annotated[
        Path,
        typer.Argument(
            help="JSON file to import.",
            exists=True,
        ),
    ],
    db: Annotated[
        Path,
        typer.Option(
            "--db",
            "-d",
            help="Path to SQLite database file.",
        ),
    ] = _DEFAULT_DB,
):
    """Import sessions from a JSON file.

    \b
    Examples:
      copilot-session-tools import-json sessions.json
      copilot-session-tools import-json exported.json --db custom.db
    """
    import json

    db.parent.mkdir(parents=True, exist_ok=True)
    database = _make_database(db)

    with json_file.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        console.print("[red]Error: JSON file must contain an array of sessions.[/red]")
        console.print("[dim]  Export first: copilot-session-tools export -o sessions.json[/dim]")
        raise typer.Exit(1)

    added = 0
    skipped = 0

    for item in data:
        if not isinstance(item, dict):
            continue

        messages = [
            ChatMessage(
                role=m.get("role", "unknown"),
                content=m.get("content", ""),
                timestamp=m.get("timestamp"),
            )
            for m in item.get("messages", [])
        ]

        session = ChatSession(
            session_id=item.get("session_id", str(hash(str(item)))),
            workspace_name=item.get("workspace_name"),
            workspace_path=item.get("workspace_path"),
            messages=messages,
            created_at=item.get("created_at"),
            updated_at=item.get("updated_at"),
            source_file=str(json_file),
            vscode_edition=item.get("vscode_edition", "imported"),
        )

        if database.add_session(session):
            added += 1
        else:
            skipped += 1

    console.print("[green]Import complete:[/green]")
    console.print(f"  Added: {added} sessions")
    console.print(f"  Skipped: {skipped} sessions")


@app.command()
def migrate(
    db: Annotated[
        Path,
        typer.Option(
            "--db",
            "-d",
            help="Path to SQLite database file.",
        ),
    ] = _DEFAULT_DB,
):
    """Run database schema migrations.

    Applies any pending schema changes to the cst_* enrichment tables.
    This is normally automatic, but can be run explicitly if the database
    was created by an older version or a migration was interrupted.

    Safe to run multiple times — already-applied migrations are skipped.

    \b
    Examples:
      copilot-session-tools migrate
      copilot-session-tools migrate --db custom.db
    """
    _ensure_db_exists(db)

    console.print(f"Migrating database: {db}")

    # Read version before migration
    import sqlite3

    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute("SELECT version FROM cst_schema_version LIMIT 1").fetchone()
        version_before = row[0] if row else 0
    except sqlite3.OperationalError:
        version_before = 0
    conn.close()

    # Force schema migration by instantiating Database
    database = _make_database(db)

    # Read version after
    from copilot_session_tools.database import CST_SCHEMA_VERSION

    conn = sqlite3.connect(str(db))
    row = conn.execute("SELECT version FROM cst_schema_version LIMIT 1").fetchone()
    version_after = row[0] if row else 0
    conn.close()

    if version_before < version_after:
        console.print(f"  [green]Migrated schema v{version_before} → v{version_after}[/green]")
    elif version_after < CST_SCHEMA_VERSION:
        console.print(f"  [yellow]Warning: schema is at v{version_after}, expected v{CST_SCHEMA_VERSION}[/yellow]")
    else:
        console.print(f"  Schema already at v{version_after} (current)")

    # Verify key columns exist
    conn = sqlite3.connect(str(db))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(cst_sessions)").fetchall()}
    conn.close()

    expected = {"enrichment_version", "parser_version", "source_format", "repository_url"}
    missing = expected - cols
    if missing:
        console.print(f"  [red]Missing columns: {', '.join(sorted(missing))}[/red]")
        console.print("  Try: copilot-session-tools scan --full")
    else:
        console.print("  [green]All expected columns present[/green]")

    # Show stats
    db_stats = database.get_stats()
    console.print(f"  Sessions: {db_stats['session_count']}, Messages: {db_stats['message_count']}")


@app.command()
def optimize(
    db: Annotated[
        Path,
        typer.Option(
            "--db",
            "-d",
            help="Path to SQLite database file.",
        ),
    ] = _DEFAULT_DB,
):
    """Optimize the full-text search index for better query performance.

    This command merges FTS5 index segments, reducing fragmentation and
    improving search speed. Recommended to run periodically, especially
    after bulk imports.

    The optimization process:
    1. Merges all FTS index segments into fewer, larger segments
    2. Runs an integrity check to verify index consistency

    \b
    Examples:
      copilot-session-tools optimize
      copilot-session-tools optimize --db custom.db
    """
    _ensure_db_exists(db)
    database = _make_database(db)

    console.print("Optimizing FTS5 search index...")

    result = database.optimize_fts()

    console.print("\n[green]Optimization complete:[/green]")
    console.print(f"  Index segments before: {result['segments_before']}")
    console.print(f"  Index segments after:  {result['segments_after']}")

    if result["segments_before"] > result["segments_after"]:
        reduction = result["segments_before"] - result["segments_after"]
        console.print(f"  [cyan]Merged {reduction} segments for faster queries[/cyan]")
    else:
        console.print("  [dim]Index was already optimized[/dim]")


@app.command()
def web(
    db: Annotated[Path, typer.Option("--db", "-d", help="Path to SQLite database file.")] = _DEFAULT_DB,
    host: Annotated[str, typer.Option("--host", "-H", help="Host to bind to.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Port to bind to.")] = 5000,
    title: Annotated[str, typer.Option("--title", "-t", help="Title for the archive.")] = "Copilot Session Tools",
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug mode.")] = False,
):
    """Start the web viewer for browsing chat sessions.

    \b
    Examples:
      copilot-session-tools web
      copilot-session-tools web --port 8080
      copilot-session-tools web --host 0.0.0.0 --port 3000
    """
    try:
        from copilot_session_tools.web import run_server
    except ImportError as err:
        console.print("[red]Web interface requires the [web] extra.[/red]")
        console.print("Install with: pip install copilot-session-tools[web]")
        raise typer.Exit(1) from err

    _ensure_db_exists(db)

    database = _make_database(str(db))
    db_stats = database.get_stats()

    if db_stats["session_count"] == 0:
        console.print("[yellow]Warning: Database is empty. Run 'copilot-session-tools scan' first.[/yellow]")

    console.print("Starting web server...")
    console.print(f"  Database: {db}")
    console.print(f"  Sessions: {db_stats['session_count']}")
    console.print(f"  Messages: {db_stats['message_count']}")
    console.print(f"\nOpen http://{host}:{port}/ in a browser to view your archive.")
    console.print("Press Ctrl+C to stop the server.\n")

    _print_upgrade_notice()

    run_server(
        host=host,
        port=port,
        db_path=str(db),
        title=title,
        debug=debug,
        chronicle_db_path=str(_chronicle_db) if _chronicle_db else None,
    )


@app.command()
def cleanup(
    session_id: str = typer.Argument(None, help="Session ID to clean up. If omitted, lists sessions with potential voice-dictated messages."),
    db: Annotated[
        Path,
        typer.Option(
            "--db",
            "-d",
            help="Path to SQLite database file.",
        ),
    ] = _DEFAULT_DB,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="LiteLLM model for cleanup (e.g., github_copilot/gpt-5.4-mini).",
        ),
    ] = "github_copilot/gpt-5.4-mini",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Preview changes without writing to the database.",
        ),
    ] = False,
    all_messages: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Clean all user messages (skip heuristic auto-detection).",
        ),
    ] = False,
    message: Annotated[
        int | None,
        typer.Option(
            "--message",
            help="Clean only this specific message index.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Re-clean already-cleaned messages.",
        ),
    ] = False,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Voice detection threshold (0.0-1.0). Lower = more sensitive.",
        ),
    ] = 0.3,
):
    """Clean up voice-dictated user messages using an LLM.

    Uses heuristics to auto-detect garbled messages, then sends them to an LLM
    (via LiteLLM) for cleanup. Original content is preserved for revert.

    Requires the [llm] extra: pip install copilot-session-tools[llm]

    \b
    Examples:
      copilot-session-tools cleanup
      copilot-session-tools cleanup a1b2c3d4-... --dry-run
      copilot-session-tools cleanup a1b2c3d4-... --all --force
      copilot-session-tools cleanup a1b2c3d4-... --message 3
    """
    try:
        from copilot_session_tools.transcript_cleanup import cleanup_session
    except ImportError:
        console.print("[red]Error: litellm is required for transcript cleanup.[/red]")
        console.print("Install with: [bold]pip install copilot-session-tools\\[llm][/bold]")
        raise typer.Exit(1)  # noqa: B904

    if session_id is None:
        _ensure_db_exists(db)
        database = _make_database(db)
        _list_cleanup_candidates(database, threshold)
        return

    _ensure_db_exists(db)
    database = _make_database(db)

    if dry_run:
        console.print("[yellow]DRY RUN — no changes will be written[/yellow]\n")

    try:
        result = cleanup_session(
            db=database,
            session_id=session_id,
            model=model,
            all_messages=all_messages,
            message_index=message,
            force=force,
            threshold=threshold,
            dry_run=dry_run,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)  # noqa: B904

    console.print(f"\n[bold]Session:[/bold] {result.session_id}")
    console.print(f"[bold]User messages:[/bold] {result.total_user_messages}")
    console.print(f"[bold]Voice-detected:[/bold] {result.detected_voice}")
    console.print(f"[bold]Cleaned:[/bold] {result.cleaned}")
    console.print(f"[bold]Skipped (already clean):[/bold] {result.skipped_clean}")
    if result.failed:
        console.print(f"[bold red]Failed:[/bold red] {result.failed}")

    for r in result.results:
        if r.is_voice and r.cleaned != r.original:
            console.print(f"\n[dim]Message {r.message_index}:[/dim]")
            console.print(f"  [red]- {r.original[:150]}{'...' if len(r.original) > 150 else ''}[/red]")
            console.print(f"  [green]+ {r.cleaned[:150]}{'...' if len(r.cleaned) > 150 else ''}[/green]")


def _list_cleanup_candidates(database: Database, threshold: float) -> None:
    """List sessions that may contain voice-dictated messages."""
    from copilot_session_tools.transcript_cleanup import compute_voice_score

    with database._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.session_id, s.custom_title, m.message_index, m.content, m.original_content
            FROM cst_messages m
            JOIN cst_sessions s ON m.session_id = s.session_id
            WHERE m.role = 'user' AND m.content IS NOT NULL AND m.agent_nesting_level = 0
            ORDER BY s.created_at DESC
            """
        )
        rows = cursor.fetchall()

    candidates: dict[str, list[tuple[int, float, str, bool]]] = {}
    for row in rows:
        sid, title, idx, content, original = row
        score = compute_voice_score(content)
        already_cleaned = original is not None
        if score >= threshold or already_cleaned:
            if sid not in candidates:
                candidates[sid] = []
            candidates[sid].append((idx, score, title or sid[:12], already_cleaned))

    if not candidates:
        console.print("[dim]No sessions with likely voice-dictated messages found.[/dim]")
        return

    console.print(f"[bold]Sessions with potential voice-dictated messages (threshold={threshold}):[/bold]\n")
    for sid, msgs in list(candidates.items())[:20]:
        title = msgs[0][2]
        cleaned_count = sum(1 for _, _, _, cleaned in msgs if cleaned)
        uncleaned = [m for m in msgs if not m[3]]
        console.print(f"  [bold]{sid[:12]}…[/bold] — {title}")
        if cleaned_count:
            console.print(f"    [green]✓ {cleaned_count} already cleaned[/green]")
        if uncleaned:
            console.print(f"    [yellow]⚠ {len(uncleaned)} candidates[/yellow] (max score: {max(s for _, s, _, _ in uncleaned):.2f})")


@app.command(name="cleanup-revert")
def cleanup_revert(
    session_id: str = typer.Argument(..., help="Session ID to revert"),
    db: Annotated[
        Path,
        typer.Option(
            "--db",
            "-d",
            help="Path to SQLite database file.",
        ),
    ] = _DEFAULT_DB,
    message: Annotated[
        int | None,
        typer.Option(
            "--message",
            help="Revert only this specific message index (otherwise reverts all).",
        ),
    ] = None,
):
    """Revert cleaned messages back to their original voice-dictated content.

    \b
    Examples:
      copilot-session-tools cleanup-revert a1b2c3d4-...
      copilot-session-tools cleanup-revert a1b2c3d4-... --message 3
    """
    from copilot_session_tools.transcript_cleanup import revert_message, revert_session

    _ensure_db_exists(db)
    database = _make_database(db)

    try:
        if message is not None:
            reverted = revert_message(database, session_id, message)
            if reverted:
                console.print(f"[green]Reverted message {message} in session {session_id}[/green]")
            else:
                console.print(f"[yellow]Message {message} has no original content to revert[/yellow]")
        else:
            count = revert_session(database, session_id)
            if count:
                console.print(f"[green]Reverted {count} message(s) in session {session_id}[/green]")
            else:
                console.print(f"[yellow]No cleaned messages found in session {session_id}[/yellow]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)  # noqa: B904


def run():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    run()
