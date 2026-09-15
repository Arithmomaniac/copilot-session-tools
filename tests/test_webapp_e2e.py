"""Playwright end-to-end tests for the webapp.

These tests are focused on client-side JavaScript behavior that cannot be tested
with Flask's test client. For HTML output validation and server-side rendering,
see test_webapp.py.

Tests that require Playwright:
- JavaScript form submission (workspace filter Apply/Clear buttons)
- Flash notification display and auto-dismiss behavior
- Client-side navigation (clicking links and verifying URL changes)
"""

import tempfile
import threading
import time
from pathlib import Path

import pytest

# Check if playwright is available
try:
    from playwright.sync_api import Page  # noqa: F401

    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

# Skip all tests in this module if playwright is not installed
pytestmark = pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="pytest-playwright not installed")

from copilot_session_tools import ChatMessage, ChatSession, Database
from copilot_session_tools.web import create_app


@pytest.fixture(scope="module")
def test_db():
    """Create a temporary database with sample data for e2e tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = Database(db_path)

    # Add session with custom title
    session1 = ChatSession(
        session_id="e2e-session-1",
        workspace_name="my-workspace",
        workspace_path="/home/user/my-workspace",
        custom_title="VS Code debug configuration for command line app",
        messages=[
            ChatMessage(role="user", content="How do I debug a command line app in VS Code?"),
            ChatMessage(
                role="assistant",
                content=(
                    "To debug a command line app in VS Code, you need to:\n\n"
                    "1. Create a launch.json file\n2. Configure the program path\n"
                    "3. Set breakpoints\n\n"
                    '```json\n{\n  "type": "python",\n  "request": "launch"\n}\n```'
                ),
            ),
            ChatMessage(role="user", content="What about Flask apps?"),
            ChatMessage(
                role="assistant",
                content=('For Flask apps, you can use the following configuration:\n\n```json\n{\n  "type": "python",\n  "module": "flask"\n}\n```'),
            ),
        ],
        created_at="1737021600000",
        vscode_edition="stable",
    )
    db.add_session(session1)

    # Add session without custom title
    session2 = ChatSession(
        session_id="e2e-session-2",
        workspace_name="another-project",
        workspace_path="/home/user/another",
        messages=[
            ChatMessage(role="user", content="What is Python?"),
            ChatMessage(
                role="assistant",
                content=("Python is a high-level, interpreted programming language known for its simplicity and readability."),
            ),
        ],
        created_at="1737108000000",
        vscode_edition="insider",
    )
    db.add_session(session2)

    yield db_path

    Path(db_path).unlink(missing_ok=True)


@pytest.fixture(scope="module")
def live_server(test_db):
    """Start a live Flask server for Playwright tests."""
    app = create_app(test_db, title="E2E Test Archive")
    app.config["TESTING"] = True

    # Run server in a thread
    server_thread = threading.Thread(target=lambda: app.run(host="127.0.0.1", port=5099, use_reloader=False, threaded=True))
    server_thread.daemon = True
    server_thread.start()

    # Wait for server to start
    time.sleep(1)

    yield "http://127.0.0.1:5099"


# NOTE: TestIndexPage, TestSessionPage, and TestErrorHandling tests have been
# moved to test_webapp.py as Flask client tests. They don't require a real
# browser since they only test HTML output, not JavaScript behavior.


class TestClearSearchNavigation:
    """Test client-side navigation for clearing search."""

    def test_clear_search(self, live_server, page):
        """Test clearing search results navigates back to index."""
        page.goto(f"{live_server}/?q=Python")

        # Click clear search link
        page.locator("a:has-text('clear search')").click()

        # Should be on index without query
        assert page.url == f"{live_server}/"

        # Both sessions should be visible
        assert page.locator("a:has-text('VS Code debug')").is_visible()
        assert page.locator("a:has-text('another-project')").is_visible()


class TestBackLinkNavigation:
    """Test client-side navigation for back links."""

    def test_back_link_navigation_from_session(self, live_server, page):
        """Test that back link navigates to index."""
        page.goto(f"{live_server}/session/e2e-session-1")

        page.locator("a.back-link[title='Back to all sessions']").click()

        # Should be on index
        assert page.url == f"{live_server}/"

    def test_back_link_navigation_from_404(self, live_server, page):
        """Test that 404 page back link navigates to index."""
        page.goto(f"{live_server}/session/nonexistent-session")

        back_link = page.locator("a:has-text('Back to all sessions'), a.back-link[title='Back to all sessions']")
        back_link.click()

        assert page.url == f"{live_server}/"


class TestWorkspaceFilter:
    """Playwright tests for workspace filter functionality."""

    def test_workspace_checkboxes_displayed(self, live_server, page):
        """Test that workspace filter checkboxes are displayed."""
        page.goto(live_server)

        # Open the workspace dropdown first
        page.locator("#workspace-dropdown-btn").click()
        page.wait_for_timeout(300)

        # Check that workspace checkboxes exist and are visible
        my_workspace_checkbox = page.locator("input[type='checkbox'][value='my-workspace']")
        another_project_checkbox = page.locator("input[type='checkbox'][value='another-project']")

        assert my_workspace_checkbox.is_visible()
        assert another_project_checkbox.is_visible()

    def test_apply_filter_button_exists(self, live_server, page):
        """Test that Apply Filter and Clear buttons exist."""
        page.goto(live_server)

        apply_btn = page.locator("button:has-text('Apply Filter')")
        clear_btn = page.locator("button:has-text('Clear')")

        assert apply_btn.is_visible()
        assert clear_btn.is_visible()

    def test_filter_by_workspace(self, live_server, page):
        """Test that selecting a workspace and applying filter works."""
        page.goto(live_server)

        # Initially both sessions should be visible
        assert page.locator("a:has-text('VS Code debug')").is_visible()
        assert page.locator("a:has-text('another-project')").is_visible()

        # Open workspace dropdown and check my-workspace
        page.locator("#workspace-dropdown-btn").click()
        page.wait_for_timeout(300)
        page.locator("input[type='checkbox'][value='my-workspace']").check()

        # Click Apply Filter
        page.locator("button:has-text('Apply Filter')").click()

        # URL should have workspace parameter
        assert "workspace=my-workspace" in page.url

        # Only my-workspace session should be visible
        assert page.locator("a:has-text('VS Code debug')").is_visible()
        assert not page.locator("a:has-text('another-project')").is_visible()

    def test_clear_filter(self, live_server, page):
        """Test that Clear button removes all filters."""
        # Start with a filter applied
        page.goto(f"{live_server}/?workspace=my-workspace")

        # Only one session should be visible
        assert page.locator("a:has-text('VS Code debug')").is_visible()
        assert not page.locator("a:has-text('another-project')").is_visible()

        # Click Clear button
        page.locator("button:has-text('Clear')").click()

        # URL should not have workspace parameter
        assert "workspace=" not in page.url

        # Both sessions should be visible
        assert page.locator("a:has-text('VS Code debug')").is_visible()
        assert page.locator("a:has-text('another-project')").is_visible()

    def test_filter_persists_in_url(self, live_server, page):
        """Test that filter state persists in URL."""
        page.goto(f"{live_server}/?workspace=another-project")

        # Checkbox should be checked
        checkbox = page.locator("input[type='checkbox'][value='another-project']")
        assert checkbox.is_checked()

        # Only another-project session should be visible
        assert page.locator("a:has-text('another-project')").is_visible()
        assert not page.locator("a:has-text('VS Code debug')").is_visible()


class TestRefreshRebuildE2E:
    """Playwright E2E tests for refresh and rebuild functionality with modified files."""

    # Class variable to track port numbers
    _port_counter = 5100

    @pytest.fixture
    def refresh_test_setup(self, tmp_path):
        """Set up a test environment with chat session files for refresh/rebuild testing."""
        import json

        # Get a unique port for this test
        TestRefreshRebuildE2E._port_counter += 1
        port = TestRefreshRebuildE2E._port_counter

        # Create database
        db_path = tmp_path / "refresh_test.db"
        db = Database(str(db_path))

        # Create workspace storage structure
        workspace_dir = tmp_path / "workspaceStorage" / "workspace1"
        chat_dir = workspace_dir / "chatSessions"
        chat_dir.mkdir(parents=True)

        # Create workspace.json
        (workspace_dir / "workspace.json").write_text(json.dumps({"folder": f"file://{tmp_path}/project1"}))

        # Create first session file
        session1_file = chat_dir / "session1.json"
        session1_file.write_text(
            json.dumps(
                {
                    "sessionId": "refresh-session-1",
                    "createdAt": "1704110400000",
                    "requests": [{"message": {"text": "Question 1"}, "timestamp": 1704110400000, "response": [{"value": "Answer 1"}]}],
                }
            )
        )

        # Create second session file
        session2_file = chat_dir / "session2.json"
        session2_file.write_text(
            json.dumps(
                {
                    "sessionId": "refresh-session-2",
                    "createdAt": "1704110500000",
                    "requests": [{"message": {"text": "Question 2"}, "timestamp": 1704110500000, "response": [{"value": "Answer 2"}]}],
                }
            )
        )

        # Import sessions into database
        from copilot_session_tools.scanner import scan_chat_sessions

        storage_paths = [(str(tmp_path / "workspaceStorage"), "stable")]
        for session in scan_chat_sessions(storage_paths, include_cli=False):
            db.add_session(session)

        return {
            "db_path": db_path,
            "db": db,
            "session1_file": session1_file,
            "session2_file": session2_file,
            "storage_paths": storage_paths,
            "tmp_path": tmp_path,
            "port": port,
        }

    @pytest.fixture
    def refresh_live_server(self, refresh_test_setup):
        """Start a live Flask server for refresh/rebuild tests."""
        db_path = refresh_test_setup["db_path"]
        storage_paths = refresh_test_setup["storage_paths"]
        port = refresh_test_setup["port"]
        app = create_app(str(db_path), title="Refresh Test Archive", storage_paths=storage_paths)
        app.config["TESTING"] = True

        # Run server in a thread
        server_thread = threading.Thread(target=lambda: app.run(host="127.0.0.1", port=port, use_reloader=False, threaded=True))
        server_thread.daemon = True
        server_thread.start()

        # Wait for server to start
        time.sleep(1)

        yield f"http://127.0.0.1:{port}"

    def test_refresh_buttons_visible(self, refresh_live_server, page):
        """Test that Refresh and Rebuild All buttons are visible."""
        page.goto(refresh_live_server)

        refresh_btn = page.locator("button:has-text('Refresh')")
        rebuild_btn = page.locator("button:has-text('Rebuild All')")

        assert refresh_btn.is_visible()
        assert rebuild_btn.is_visible()

    def test_refresh_shows_notification(self, refresh_live_server, page):
        """Test that clicking Refresh shows a notification."""
        page.goto(refresh_live_server)

        # Click Refresh button
        page.locator("button:has-text('🔄 Refresh')").click()

        # Should show notification
        notification = page.locator(".refresh-notification")
        assert notification.is_visible()
        assert "Incremental refresh complete" in notification.text_content()

    def test_rebuild_shows_notification(self, refresh_live_server, page):
        """Test that clicking Rebuild All shows a notification."""
        page.goto(refresh_live_server)

        # Click Rebuild All button
        page.locator("button:has-text('🔨 Rebuild All')").click()

        # Should show notification
        notification = page.locator(".refresh-notification")
        assert notification.is_visible()
        assert "Full refresh complete" in notification.text_content()

    def test_refresh_skips_unchanged_files(self, refresh_test_setup, refresh_live_server, page):
        """Test that Refresh correctly skips unchanged files."""
        page.goto(refresh_live_server)

        # Click Refresh without modifying any files
        page.locator("button:has-text('🔄 Refresh')").click()

        # Should show notification with skipped sessions
        notification = page.locator(".refresh-notification")
        assert notification.is_visible()
        text = notification.text_content()

        # All sessions should be skipped (none added, none updated)
        assert "Incremental refresh complete" in text
        assert "Added 0" in text
        assert "updated 0" in text
        assert "skipped 2" in text  # Both sessions unchanged

    def test_refresh_updates_only_modified_file(self, refresh_test_setup, refresh_live_server, page):
        """Test that Refresh only updates the modified file, not unchanged ones."""
        import json

        # Modify only one session file
        time.sleep(0.1)  # Ensure mtime changes
        refresh_test_setup["session1_file"].write_text(
            json.dumps(
                {
                    "sessionId": "refresh-session-1",
                    "createdAt": "1704110400000",
                    "requests": [
                        {"message": {"text": "Question 1"}, "timestamp": 1704110400000, "response": [{"value": "Answer 1"}]},
                        {"message": {"text": "Added question"}, "timestamp": 1704110600000, "response": [{"value": "Added answer"}]},
                    ],
                }
            )
        )

        page.goto(refresh_live_server)

        # Click Refresh
        page.locator("button:has-text('🔄 Refresh')").click()

        # Should show notification
        notification = page.locator(".refresh-notification")
        assert notification.is_visible()
        text = notification.text_content()

        # Only 1 session should be updated, 1 should be skipped
        assert "Incremental refresh complete" in text
        assert "Added 0" in text
        assert "updated 1" in text
        assert "skipped 1" in text

    def test_rebuild_updates_all_files(self, refresh_test_setup, refresh_live_server, page):
        """Test that Rebuild All updates all files, even unchanged ones."""
        page.goto(refresh_live_server)

        # Click Rebuild All
        page.locator("button:has-text('🔨 Rebuild All')").click()

        # Should show notification with updated sessions (none added because they exist)
        notification = page.locator(".refresh-notification")
        assert notification.is_visible()
        text = notification.text_content()

        # All sessions should be updated (not skipped)
        assert "Full refresh complete" in text
        assert "skipped 0" in text  # Nothing skipped in full mode

    def test_refresh_vs_rebuild_different_behavior_with_modified_file(self, refresh_test_setup, page):
        """Compare Refresh vs Rebuild behavior when one file is modified."""
        import json

        db_path = refresh_test_setup["db_path"]
        storage_paths = refresh_test_setup["storage_paths"]

        # Use the port from setup + 50 to avoid conflicts
        port = refresh_test_setup["port"] + 50

        # Start a fresh server with custom storage paths
        app = create_app(str(db_path), title="Compare Test", storage_paths=storage_paths)
        app.config["TESTING"] = True

        server_thread = threading.Thread(target=lambda: app.run(host="127.0.0.1", port=port, use_reloader=False, threaded=True))
        server_thread.daemon = True
        server_thread.start()
        time.sleep(1)

        server_url = f"http://127.0.0.1:{port}"

        # First, do a refresh with no changes - both should be skipped
        page.goto(server_url)
        page.locator("button:has-text('🔄 Refresh')").click()

        notification = page.locator(".refresh-notification")
        initial_refresh_text = notification.text_content()
        assert "skipped 2" in initial_refresh_text

        # Modify one file
        time.sleep(0.1)
        refresh_test_setup["session1_file"].write_text(
            json.dumps(
                {
                    "sessionId": "refresh-session-1",
                    "createdAt": "1704110400000",
                    "requests": [{"message": {"text": "Modified question"}, "timestamp": 1704110400000, "response": [{"value": "Modified answer"}]}],
                }
            )
        )

        # Refresh should update only the modified file
        page.goto(server_url)
        page.locator("button:has-text('🔄 Refresh')").click()

        notification = page.locator(".refresh-notification")
        refresh_text = notification.text_content()
        assert "Incremental refresh complete" in refresh_text
        assert "updated 1" in refresh_text
        assert "skipped 1" in refresh_text  # One file unchanged

        # Now Rebuild All should update all files (no skipped)
        page.goto(server_url)
        page.locator("button:has-text('🔨 Rebuild All')").click()

        notification = page.locator(".refresh-notification")
        rebuild_text = notification.text_content()
        assert "Full refresh complete" in rebuild_text
        assert "skipped 0" in rebuild_text  # Full mode doesn't skip

    def test_notification_disappears_on_page_reload(self, refresh_live_server, page):
        """Test that the notification disappears after page reload (flash behavior)."""
        page.goto(refresh_live_server)

        # Click Refresh
        page.locator("button:has-text('🔄 Refresh')").click()

        # Should show notification
        assert page.locator(".refresh-notification").is_visible()

        # Reload page
        page.reload()

        # Notification should be gone
        assert not page.locator(".refresh-notification").is_visible()
