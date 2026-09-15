"""Flask web application for viewing Copilot chat archive."""

import json
import re
import secrets

from flask import Flask, flash, jsonify, make_response, redirect, render_template, request, session, url_for

from copilot_session_tools import Database, __version__, generate_session_filename, get_vscode_storage_paths
from copilot_session_tools.content_types import SEARCH_CONTENT_TYPES, resolve_search_content_set
from copilot_session_tools.html_exporter import generate_session_html_filename, session_to_html
from copilot_session_tools.refresh import enrich_single_session, run_enrichment, run_refresh
from copilot_session_tools.sources import SessionIdentityConflictError, discover_source_presets
from copilot_session_tools.utils import (
    build_block_metadata,
    build_root_agent_markers,
    detect_language,
    extract_filename,
    format_timestamp,
    highlight_code,
    markdown_to_html,
    match_tool_for_block,
    parse_diff_stats,
    prettify_json,
    strip_ansi,
    truncate_preview,
    urldecode,
)


def _from_json(value: str) -> dict:
    """Jinja2 filter: parse a JSON string into a dict."""
    try:
        return json.loads(value) if value else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def create_app(
    db_path: str,
    title: str = "Copilot Session Tools",
    storage_paths: list | None = None,
    chronicle_db_path: str | None = None,
) -> Flask:
    """Create and configure the Flask application.

    Args:
        db_path: Path to the CST enrichment database file.
        title: Title for the archive.
        storage_paths: Optional list of (path, edition) tuples for scanning.
                       If None, uses default VS Code storage paths.
        chronicle_db_path: Path to the Copilot CLI Chronicle session-store.db.
                          If None, auto-detects as sibling of db_path.

    Returns:
        Configured Flask application.
    """
    app = Flask(
        __name__,
        template_folder="templates",
    )

    # Set a secret key for session support (used for transient flash messages)
    # A random key is fine here since sessions only contain ephemeral refresh notifications.
    # Set FLASK_SECRET_KEY environment variable for persistent sessions across restarts.
    import os

    app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

    # Register Jinja2 filters
    app.jinja_env.filters["markdown"] = markdown_to_html
    app.jinja_env.filters["urldecode"] = urldecode
    app.jinja_env.filters["format_timestamp"] = format_timestamp
    app.jinja_env.filters["parse_diff_stats"] = parse_diff_stats
    app.jinja_env.filters["extract_filename"] = extract_filename
    app.jinja_env.filters["strip_ansi"] = strip_ansi
    app.jinja_env.filters["truncate_preview"] = truncate_preview
    app.jinja_env.filters["from_json"] = _from_json
    app.jinja_env.filters["prettify_json"] = prettify_json

    # Register global functions for template calls
    app.jinja_env.globals["detect_language"] = detect_language
    app.jinja_env.globals["highlight_code"] = highlight_code

    # Register global function for tool matching
    app.jinja_env.globals["match_tool_for_block"] = match_tool_for_block
    app.jinja_env.globals["get_nested_meta"] = lambda block: getattr(block, "_nested_meta", {})

    # Store database path, title, storage paths, and Chronicle path in app config
    app.config["DB_PATH"] = db_path
    app.config["CHRONICLE_DB_PATH"] = chronicle_db_path
    app.config["ARCHIVE_TITLE"] = title
    app.config["STORAGE_PATHS"] = storage_paths  # None means use default VS Code paths

    # Check for PyPI upgrade (cached, non-blocking)
    try:
        from copilot_session_tools.version_check import check_for_upgrade

        app.config["UPGRADE_AVAILABLE"] = check_for_upgrade()
    except Exception:
        app.config["UPGRADE_AVAILABLE"] = None

    def _create_snippet(content: str, max_length: int = 150) -> str:
        """Create a snippet from content, normalizing whitespace."""
        if not content:
            return ""
        # Normalize whitespace (replace newlines and multiple spaces with single space)

        normalized = re.sub(r"\s+", " ", content).strip()
        if len(normalized) > max_length:
            return normalized[:max_length] + "..."
        return normalized

    def _csrf_token() -> str:
        token = session.get("csrf_token")
        if not token:
            token = secrets.token_urlsafe(32)
            session["csrf_token"] = token
        return token

    def _require_csrf() -> None:
        if app.config.get("TESTING"):
            return
        expected = session.get("csrf_token", "")
        provided = request.form.get("csrf_token", "")
        if not expected or not secrets.compare_digest(expected, provided):
            from flask import abort

            abort(400, "Invalid CSRF token.")

    @app.errorhandler(SessionIdentityConflictError)
    def handle_session_identity_conflict(error: SessionIdentityConflictError):
        return make_response(str(error), 409)

    @app.route("/")
    def index():
        """List sessions, with optional search, workspace, repository filtering, and pagination."""
        db = Database(app.config["DB_PATH"], chronicle_db_path=app.config["CHRONICLE_DB_PATH"])
        query = request.args.get("q", "").strip()
        selected_workspaces = request.args.getlist("workspace")
        selected_repositories = request.args.getlist("repository")
        selected_editions = request.args.getlist("edition")
        selected_source = request.args.get("source", "").strip()
        selected_search_types = request.args.getlist("search_in")
        sort_by = request.args.get("sort", "relevance")  # 'relevance' or 'date'

        # Pagination parameters
        try:
            page = max(1, int(request.args.get("page", 1)))
        except (ValueError, TypeError):
            page = 1
        per_page = 20  # Sessions per page

        # Get refresh results from session (set after a refresh operation)
        # Pop to ensure it's only shown once
        refresh_result = session.pop("refresh_result", None)

        search_snippets = {}  # session_id -> list of snippets with message links

        # Build content type filter from search_in params
        # Use resolve_search_content_set for parent-child auto-inclusion
        # (e.g., selecting only "diffs" must auto-include "file-changes")
        search_content_set = None  # None = search everything (default)
        if selected_search_types:
            valid_types = set(SEARCH_CONTENT_TYPES.keys())
            validated = [t for t in selected_search_types if t in valid_types]
            if validated:
                try:
                    search_content_set = resolve_search_content_set(include=validated)
                except Exception:
                    search_content_set = None  # Fallback to defaults on error

        sources = db.list_sources()
        all_sessions = db.list_sessions()
        sessions_by_id = {item["session_id"]: item for item in all_sessions}
        application_filters: list[dict] = []
        application_sessions: dict[str, list[dict]] = {}

        edition_names: set[str] = set()
        for item in all_sessions:
            edition_name = item.get("vscode_edition")
            if isinstance(edition_name, str) and edition_name != "cli":
                edition_names.add(edition_name)
        for edition_name in sorted(edition_names):
            matching = [item for item in all_sessions if item.get("vscode_edition") == edition_name]
            application_sessions[edition_name] = matching
            application_filters.append(
                {
                    "key": edition_name,
                    "label": edition_name,
                    "css_class": edition_name,
                    "count": len(matching),
                }
            )

        for source in sources:
            key = "cli" if source.is_default else f"source:{source.source_id}"
            source_session_ids = db.get_source_session_ids(source)
            matching = []
            for session_id in source_session_ids:
                if session_id not in sessions_by_id:
                    continue
                item = dict(sessions_by_id[session_id])
                item["source_id"] = source.source_id
                item["source_name"] = source.name
                matching.append(item)
            application_sessions[key] = matching
            application_filters.append(
                {
                    "key": key,
                    "label": "cli" if source.is_default else source.name,
                    "css_class": "cli",
                    "count": len(matching),
                }
            )

        if selected_source and not selected_editions:
            legacy_source = db.get_source(selected_source)
            if legacy_source is not None:
                selected_editions = ["cli" if legacy_source.is_default else f"source:{legacy_source.source_id}"]
                selected_source = ""

        if selected_editions:
            selected_session_map: dict[str, dict] = {}
            for application_key in selected_editions:
                for item in application_sessions.get(application_key, []):
                    selected_session_map.setdefault(item["session_id"], item)
            candidate_sessions = sorted(
                selected_session_map.values(),
                key=lambda item: item.get("updated_at") or "",
                reverse=True,
            )
        else:
            candidate_sessions = all_sessions

        if query:
            # Use FTS search with sort option
            # The db.search() returns results in the correct order based on sort_by
            workspace_query = " ".join(f'workspace:"{workspace.replace(chr(34), chr(34) + chr(34))}"' for workspace in selected_workspaces)
            search_query = f"{query} {workspace_query}".strip()
            search_results = db.search(
                search_query,
                limit=100,
                sort_by=sort_by,
                search_content_set=search_content_set,
                repository=selected_repositories or None,
            )

            # Group results by session and collect snippets
            # session_ids preserves the order from search results (for relevance sorting)
            session_ids = []
            for r in search_results:
                sid = r["session_id"]
                if sid not in search_snippets:
                    search_snippets[sid] = []
                    session_ids.append(sid)

                # Add snippet (up to 5 per session, each from different message)
                if len(search_snippets[sid]) < 5:
                    # message_index is 0-based, but anchor is 1-based
                    msg_index = r.get("message_index", 0)
                    snippet = {
                        "text": _create_snippet(r.get("highlighted", r.get("content", ""))),
                        "message_anchor": msg_index + 1,  # 1-based for #msg-N
                        "active_context": r.get("active_context"),
                    }
                    search_snippets[sid].append(snippet)

            # Get full session info for matching sessions, preserving search result order
            session_map = {s["session_id"]: s for s in candidate_sessions}
            sessions = [session_map[sid] for sid in session_ids if sid in session_map]
        else:
            # No query: list_sessions() returns sessions sorted by date (newest first)
            # Relevance sorting doesn't apply without a search query
            sessions = candidate_sessions

        # Apply workspace filter if selected
        if selected_workspaces:
            selected_workspace_set = set(selected_workspaces)
            sessions = [s for s in sessions if selected_workspace_set.intersection(set(s.get("workspace_names") or [s.get("workspace_name")]))]

        # Apply repository filter if selected
        if selected_repositories:
            selected_repository_set = set(selected_repositories)
            sessions = [s for s in sessions if selected_repository_set.intersection(set(s.get("repository_urls") or [s.get("repository_url")]))]

        # Calculate pagination
        total_sessions = len(sessions)
        total_pages = max(1, (total_sessions + per_page - 1) // per_page)  # Ceiling division
        page = min(page, total_pages)  # Ensure page doesn't exceed total
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_sessions = sessions[start_idx:end_idx]

        workspaces = db.get_workspaces(all_sessions)
        repositories = db.get_repositories(all_sessions)
        stats = db.get_stats(all_sessions)
        cst_tables_exist = db.has_cst_tables()
        version_refresh_count = db.count_sessions_needing_version_refresh(__version__) if cst_tables_exist else 0

        return render_template(
            "index.html",
            title=app.config["ARCHIVE_TITLE"],
            app_version=__version__,
            sessions=paginated_sessions,
            workspaces=workspaces,
            repositories=repositories,
            stats=stats,
            query=query,
            search_snippets=search_snippets,
            selected_workspaces=selected_workspaces,
            selected_repositories=selected_repositories,
            selected_editions=selected_editions,
            application_filters=application_filters,
            selected_search_types=selected_search_types,
            refresh_result=refresh_result,
            sort_by=sort_by,
            has_cst_tables=cst_tables_exist,
            version_refresh_count=version_refresh_count,
            upgrade_available=app.config.get("UPGRADE_AVAILABLE"),
            csrf_token=_csrf_token(),
            # Pagination context
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            total_sessions=total_sessions,
        )

    @app.route("/session/<session_id>")
    def session_view(session_id: str):
        """Render a single session."""
        db = Database(app.config["DB_PATH"], chronicle_db_path=app.config["CHRONICLE_DB_PATH"])
        session = db.get_session(session_id)

        if session is None:
            return render_template(
                "error.html",
                title=app.config["ARCHIVE_TITLE"],
                error="Session not found",
                message=f"No session found with ID: {session_id}",
            ), 404

        # Determine if this session has enriched (cst_*) data
        is_enriched = False
        if db.has_cst_tables():
            with db._get_connection() as conn:
                row = conn.execute("SELECT 1 FROM cst_sessions WHERE session_id = ?", (session_id,)).fetchone()
                is_enriched = row is not None

        # For unenriched sessions, build simple turns list for the template
        turns = None
        new_turns = None
        if not is_enriched:
            turns = db.get_builtin_turns(session_id)
        else:
            # Check for new turns added since last enrichment
            enriched_turn_count = sum(1 for m in session.messages if m.role == "user")
            all_builtin_turns = db.get_builtin_turns(session_id)
            if len(all_builtin_turns) > enriched_turn_count:
                new_turns = all_builtin_turns[enriched_turn_count:]

        # Pre-process messages to match tool invocations and command runs with content blocks
        # This creates a mapping that the template can use directly
        first_user_prompt = None
        message_metadata: dict[int, dict] = {}
        for msg_idx, message in enumerate(session.messages):
            # Capture first user prompt for title fallback
            if message.role == "user" and first_user_prompt is None:
                first_user_prompt = message.content

            message_metadata[msg_idx] = build_block_metadata(message.content_blocks, message.tool_invocations, message.command_runs)

        # Reconstruct IO entries for async shell blocks from shell backlink data.
        # IO entries are not stored in the DB — they're rebuilt at render time by
        # matching shell backlinks (read/write/stop_powershell) to their parent CommandRun.
        from copilot_session_tools.scanner.models import ShellIOEntry

        shell_cmd_runs: dict[str, list[tuple[int, object]]] = {}
        for msg_idx, message in enumerate(session.messages):
            for cmd in message.command_runs:
                if cmd.is_async and cmd.shell_id:
                    shell_cmd_runs.setdefault(cmd.shell_id, []).append((msg_idx, cmd))
        if shell_cmd_runs:
            io_counters: dict[str, int] = {}
            for msg_idx, message in enumerate(session.messages):
                for tool in message.tool_invocations:
                    if not tool.is_shell_backlink or not tool.backlink_shell_id:
                        continue
                    runs = shell_cmd_runs.get(tool.backlink_shell_id)
                    if not runs:
                        continue
                    parent_cmd = None
                    for run_msg_idx, cmd in reversed(runs):
                        if run_msg_idx <= msg_idx:
                            parent_cmd = cmd
                            break
                    if not parent_cmd:
                        parent_cmd = runs[0][1]
                    action = tool.name.replace("_powershell", "")
                    counter_key = f"{tool.backlink_shell_id}-{id(parent_cmd)}-{action}"
                    io_counters[counter_key] = io_counters.get(counter_key, 0) + 1
                    count = io_counters[counter_key]
                    anchor_id = f"io-{tool.backlink_shell_id}-{action}-{count}"
                    pill_id = f"pill-{tool.backlink_shell_id}-{action}-{count}"
                    input_text = None
                    if action == "write" and tool.input:
                        try:
                            args = json.loads(tool.input)
                            input_text = args.get("input", "")
                        except (json.JSONDecodeError, TypeError):
                            pass
                    parent_cmd.io_entries.append(
                        ShellIOEntry(
                            action=action,
                            input_text=input_text,
                            result=tool.result,
                            anchor_id=anchor_id,
                            pill_id=pill_id,
                        )
                    )
                    tool.shell_pill_id = pill_id
                    tool.shell_anchor_id = anchor_id

        enrichment_version = db.get_session_enrichment_version(session_id) if is_enriched else None
        needs_version_refresh = False
        if is_enriched and (enrichment_version is None or enrichment_version != __version__):
            from packaging.version import Version

            needs_version_refresh = enrichment_version is None or Version(enrichment_version) < Version(__version__)

        # Check if any messages have been cleaned (for showing revert-all button)
        has_cleanup_data = any(m.original_content is not None for m in session.messages) if is_enriched else False

        return render_template(
            "session.html",
            title=app.config["ARCHIVE_TITLE"],
            app_version=__version__,
            session=session,
            message_count=len(session.messages),
            first_user_prompt=first_user_prompt,
            message_metadata=message_metadata,
            root_agent_markers=build_root_agent_markers(session) if is_enriched else {},
            is_enriched=is_enriched,
            turns=turns,
            new_turns=new_turns,
            needs_version_refresh=needs_version_refresh,
            enrichment_version=enrichment_version,
            has_cleanup_data=has_cleanup_data,
        )

    @app.route("/refresh", methods=["POST"])
    def refresh_database():
        """Refresh the database by scanning for new or updated sessions.

        Supports two modes:
        - full=false (default): Incremental refresh, only updates changed sessions
        - full=true: Full rebuild, re-imports all sessions
        """
        db = Database(app.config["DB_PATH"], chronicle_db_path=app.config["CHRONICLE_DB_PATH"])
        full_refresh = request.form.get("full", "false").lower() == "true"

        # Get storage paths - use configured paths or default VS Code paths
        # Check for None explicitly since empty list [] is a valid value (for testing)
        storage_paths = app.config.get("STORAGE_PATHS")
        if storage_paths is None:
            storage_paths = get_vscode_storage_paths()

        result = run_refresh(db, storage_paths, full=full_refresh, workers=1)
        enrich_result = run_enrichment(db, workers=1)

        # Store refresh result in Flask session for display after redirect
        session["refresh_result"] = {
            "added": result.added,
            "updated": result.updated,
            "skipped": result.skipped,
            "mode": result.mode,
            "enriched": enrich_result.enriched,
            "reparsed": enrich_result.reparsed,
        }

        return redirect(url_for("index"))

    @app.route("/sources")
    def sources():
        """Manage named Copilot CLI application sources."""
        db = Database(app.config["DB_PATH"], chronicle_db_path=app.config["CHRONICLE_DB_PATH"])
        return render_template(
            "sources.html",
            title=app.config["ARCHIVE_TITLE"],
            sources=db.list_sources(),
            presets=discover_source_presets(),
            csrf_token=_csrf_token(),
        )

    @app.route("/sources/add", methods=["POST"])
    def add_source():
        _require_csrf()
        db = Database(app.config["DB_PATH"], chronicle_db_path=app.config["CHRONICLE_DB_PATH"])
        base_dir = request.form.get("base_dir", "").strip()
        preset_id = request.form.get("preset", "").strip()
        if bool(base_dir) == bool(preset_id):
            flash("Specify exactly one base directory or detected preset.", "error")
            return redirect(url_for("sources"))
        if preset_id:
            presets = {preset.preset_id: preset for preset in discover_source_presets()}
            preset = presets.get(preset_id)
            if preset is None:
                flash(f"Source preset is not available: {preset_id}", "error")
                return redirect(url_for("sources"))
            base_dir = str(preset.base_dir)
        try:
            source = db.add_source(request.form.get("name", ""), base_dir)
        except ValueError as exc:
            flash(str(exc), "error")
        else:
            flash(f"Added source {source.name}.", "success")
        return redirect(url_for("sources"))

    @app.route("/sources/<source_id>/rename", methods=["POST"])
    def rename_source(source_id: str):
        _require_csrf()
        db = Database(app.config["DB_PATH"], chronicle_db_path=app.config["CHRONICLE_DB_PATH"])
        try:
            source = db.rename_source(source_id, request.form.get("name", ""))
        except ValueError as exc:
            flash(str(exc), "error")
        else:
            flash(f"Renamed source to {source.name}.", "success")
        return redirect(url_for("sources"))

    @app.route("/sources/<source_id>/toggle", methods=["POST"])
    def toggle_source(source_id: str):
        _require_csrf()
        db = Database(app.config["DB_PATH"], chronicle_db_path=app.config["CHRONICLE_DB_PATH"])
        enabled = request.form.get("enabled") == "true"
        try:
            source = db.set_source_enabled(source_id, enabled=enabled)
        except ValueError as exc:
            flash(str(exc), "error")
        else:
            flash(f"{'Enabled' if source.enabled else 'Disabled'} source {source.name}.", "success")
        return redirect(url_for("sources"))

    @app.route("/enrich/<session_id>", methods=["POST"])
    def enrich_session(session_id: str):
        """Enrich a single CLI session by parsing its events.jsonl file."""
        enrich_db = Database(app.config["DB_PATH"], chronicle_db_path=app.config["CHRONICLE_DB_PATH"])
        try:
            existing = enrich_db.get_session(session_id)
        except SessionIdentityConflictError as exc:
            flash(str(exc), "error")
            return redirect(url_for("session_view", session_id=session_id))
        if existing is None:
            flash("Session not found.", "error")
            return redirect(url_for("session_view", session_id=session_id))
        source = enrich_db.get_source(existing.source_id) if existing.source_id else None
        if source is None or not source.enabled:
            flash("The session source is unavailable or disabled.", "error")
            return redirect(url_for("session_view", session_id=session_id))
        error = enrich_single_session(
            enrich_db,
            existing.native_session_id or existing.session_id,
            source=source,
        )
        if error:
            flash(error, "error")
            return redirect(url_for("session_view", session_id=session_id))
        return redirect(url_for("session_view", session_id=session_id))

    @app.route("/api/cleanup/<session_id>", methods=["POST"])
    def api_cleanup_session(session_id: str):
        """Clean up voice-dictated messages in a session using LLM."""
        try:
            from copilot_session_tools.transcript_cleanup import cleanup_session
        except ImportError:
            return jsonify({"success": False, "error": "litellm not installed. Install with: pip install copilot-session-tools[llm]"}), 500

        cleanup_db = Database(app.config["DB_PATH"], chronicle_db_path=app.config["CHRONICLE_DB_PATH"])
        data = request.get_json(silent=True) or {}
        message_index = data.get("message_index")

        try:
            result = cleanup_session(
                db=cleanup_db,
                session_id=session_id,
                message_index=message_index,
                all_messages=message_index is None,
            )
            return jsonify(
                {
                    "success": True,
                    "cleaned": result.cleaned,
                    "detected_voice": result.detected_voice,
                    "skipped_clean": result.skipped_clean,
                    "results": [{"index": r.message_index, "is_voice": r.is_voice, "original": r.original, "cleaned": r.cleaned} for r in result.results],
                }
            )
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/cleanup-revert/<session_id>", methods=["POST"])
    def api_cleanup_revert(session_id: str):
        """Revert cleaned messages to their original content."""
        from copilot_session_tools.transcript_cleanup import revert_message, revert_session

        revert_db = Database(app.config["DB_PATH"], chronicle_db_path=app.config["CHRONICLE_DB_PATH"])
        data = request.get_json(silent=True) or {}
        message_index = data.get("message_index")

        try:
            if message_index is not None:
                reverted = revert_message(revert_db, session_id, message_index)
                return jsonify({"success": True, "reverted_count": 1 if reverted else 0})
            else:
                count = revert_session(revert_db, session_id)
                return jsonify({"success": True, "reverted_count": count})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/markdown/<session_id>", methods=["GET"])
    def get_markdown(session_id: str):
        """Get markdown for a session's messages.

        Query parameters:
        - start: Start message number (1-based, optional, defaults to 1)
        - end: End message number (1-based, optional, defaults to last message)
        - content_set: Comma-separated content types to include (e.g. "diffs,thinking")
        - download: If true, return as a file attachment instead of JSON (default: false)

        Legacy query parameters (used if content_set is absent):
        - include_diffs, include_tool_inputs, include_thinking, include_agent_details

        Returns:
            JSON with 'markdown' field, or a .md file download if download=true.
        """
        from copilot_session_tools.content_types import DEFAULT_INCLUDES

        db = Database(app.config["DB_PATH"], chronicle_db_path=app.config["CHRONICLE_DB_PATH"])

        # Parse range parameters
        start_param = request.args.get("start", "").strip()
        end_param = request.args.get("end", "").strip()

        # Build content_set — prefer explicit param, fall back to legacy booleans
        if "content_set" in request.args:
            content_set_param = request.args.get("content_set", "").strip()
            content_set = {t.strip() for t in content_set_param.split(",") if t.strip()} if content_set_param else set()
        else:
            # Backward compat: build content_set from legacy boolean params
            content_set = set(DEFAULT_INCLUDES)
            if request.args.get("include_diffs", "true").lower() == "true":
                content_set.add("diffs")
            if request.args.get("include_tool_inputs", "true").lower() == "true":
                content_set.add("tool-inputs")
            if request.args.get("include_thinking", "false").lower() == "true":
                content_set.add("thinking")
            if request.args.get("include_agent_details", "true").lower() != "true":
                content_set.discard("agent-details")

        download = request.args.get("download", "false").lower() == "true"

        start = None
        end = None

        if start_param:
            try:
                start = int(start_param)
            except ValueError:
                return jsonify({"error": "Invalid start value"}), 400

        if end_param:
            try:
                end = int(end_param)
            except ValueError:
                return jsonify({"error": "Invalid end value"}), 400

        if start is not None and start < 1:
            return jsonify({"error": "start must be >= 1"}), 400
        if end is not None and end < 1:
            return jsonify({"error": "end must be >= 1"}), 400
        if start is not None and end is not None and start > end:
            return jsonify({"error": "start must be <= end"}), 400

        # For download mode, we need the session object for filename generation
        chat_session = None
        if download:
            chat_session = db.get_session(session_id)
            if chat_session is None:
                return jsonify({"error": "Session not found"}), 404

        markdown_content = db.get_messages_markdown(
            session_id,
            start=start,
            end=end,
            content_set=content_set,
        )

        if not markdown_content:
            return jsonify({"error": "No messages found"}), 404

        if download and chat_session is not None:
            filename = generate_session_filename(chat_session)
            response = make_response(markdown_content)
            response.headers["Content-Type"] = "text/markdown; charset=utf-8"
            response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
            return response

        return jsonify({"markdown": markdown_content})

    @app.route("/api/html/<session_id>", methods=["GET"])
    def api_html(session_id: str):
        """Export session as self-contained static HTML file.

        Query parameters:
        - start: Start message number (1-based, optional)
        - end: End message number (1-based, optional)
        - content_set: Comma-separated content types to include (e.g. "diffs,thinking")

        Returns:
            HTML file download.
        """
        db = Database(app.config["DB_PATH"], chronicle_db_path=app.config["CHRONICLE_DB_PATH"])

        # Parse content_set
        if "content_set" in request.args:
            content_set_param = request.args.get("content_set", "").strip()
            content_set: set[str] | None = {t.strip() for t in content_set_param.split(",") if t.strip()} if content_set_param else set()
        else:
            content_set = None  # Use defaults

        # Parse message range
        start_param = request.args.get("start", "").strip()
        end_param = request.args.get("end", "").strip()

        start = None
        end = None

        if start_param:
            try:
                start = int(start_param)
            except ValueError:
                return jsonify({"error": "Invalid start value"}), 400

        if end_param:
            try:
                end = int(end_param)
            except ValueError:
                return jsonify({"error": "Invalid end value"}), 400

        if start is not None and start < 1:
            return jsonify({"error": "start must be >= 1"}), 400
        if end is not None and end < 1:
            return jsonify({"error": "end must be >= 1"}), 400
        if start is not None and end is not None and start > end:
            return jsonify({"error": "start must be <= end"}), 400

        # Get session from database
        chat_session = db.get_session(session_id)
        if chat_session is None:
            return jsonify({"error": "Session not found"}), 404

        # Apply message range if specified
        if start is not None or end is not None:
            messages = chat_session.messages
            start_idx = (start - 1) if start else 0
            end_idx = end if end else len(messages)
            chat_session.messages = messages[start_idx:end_idx]

        html_content = session_to_html(chat_session, content_set=content_set)
        filename = generate_session_html_filename(chat_session)

        response = make_response(html_content)
        response.headers["Content-Type"] = "text/html; charset=utf-8"
        response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 5000,
    db_path: str = "copilot_chats.db",
    title: str = "Copilot Session Tools",
    debug: bool = False,
    chronicle_db_path: str | None = None,
) -> None:
    """Run the Flask development server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        db_path: Path to the CST enrichment database file.
        title: Title for the archive.
        debug: Enable debug mode.
        chronicle_db_path: Path to the Chronicle session-store.db (auto-detected if None).
    """
    app = create_app(db_path, title, chronicle_db_path=chronicle_db_path)
    # Suppress Flask/Werkzeug startup banners — we print our own
    import logging

    logging.getLogger("werkzeug").setLevel(logging.WARNING if debug else logging.ERROR)
    app.run(host=host, port=port, debug=debug)
