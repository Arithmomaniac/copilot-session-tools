"""Named Copilot CLI source definitions and validation."""

from __future__ import annotations

import os
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SOURCE_ID = "default-cli"
DEFAULT_SOURCE_NAME = "cli"
RESERVED_SOURCE_NAMES = frozenset({"cli", "stable", "insider", "custom", "copilot cli"})


class SessionIdentityConflictError(ValueError):
    """Raised when different session data is found under the same UUID."""


@dataclass(frozen=True)
class CopilotSource:
    """A Copilot CLI application's persisted local data source."""

    source_id: str
    name: str
    base_dir: Path
    enabled: bool
    is_default: bool = False

    @property
    def chronicle_db_path(self) -> Path:
        return self.base_dir / "session-store.db"


@dataclass(frozen=True)
class SourcePreset:
    """A detected application source that can be registered explicitly."""

    preset_id: str
    name: str
    base_dir: Path


def normalize_source_name(name: str) -> tuple[str, str]:
    """Return a canonical lowercase identifier and uniqueness key."""
    identifier = name.strip()
    if not identifier:
        raise ValueError("Source name must not be empty.")
    if any(ord(character) < 32 for character in identifier):
        raise ValueError("Source name must not contain control characters.")
    identifier = identifier.casefold()
    if identifier in RESERVED_SOURCE_NAMES:
        raise ValueError(f"Source name '{identifier}' is reserved.")
    return identifier, identifier


def canonicalize_base_dir(base_dir: str | Path) -> tuple[Path, str]:
    """Resolve a source directory and return its filesystem uniqueness key."""
    path = Path(base_dir).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Copilot base directory is unavailable: {path}") from exc
    if not resolved.is_dir():
        raise ValueError(f"Copilot base directory is not a directory: {resolved}")
    return resolved, os.path.normcase(str(resolved))


def validate_chronicle(base_dir: Path) -> None:
    """Verify that a base directory contains a readable Chronicle database."""
    db_path = base_dir / "session-store.db"
    if not db_path.is_file():
        raise ValueError(f"Chronicle database not found: {db_path}")
    try:
        with closing(sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('sessions', 'turns')")}
    except sqlite3.Error as exc:
        raise ValueError(f"Chronicle database is not readable: {db_path}") from exc
    missing = {"sessions", "turns"} - tables
    if missing:
        raise ValueError(f"Chronicle database is missing required tables: {', '.join(sorted(missing))}")


def new_source_id() -> str:
    """Generate an opaque immutable source identifier."""
    return uuid.uuid4().hex


def discover_source_presets(home: Path | None = None) -> list[SourcePreset]:
    """Return locally available application presets without registering them."""
    home_dir = home or Path.home()
    scout_dir = home_dir / ".scout" / "copilot"
    if (scout_dir / "session-store.db").is_file():
        return [SourcePreset(preset_id="scout", name="scout", base_dir=scout_dir)]
    return []
