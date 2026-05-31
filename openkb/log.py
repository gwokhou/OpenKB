"""Append-only operation log for the wiki (log.md)."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from openkb.locks import maybe_kb_ingest_lock


def append_log(
    wiki_dir: Path, operation: str, description: str, *, assume_locked: bool = False
) -> None:
    """Append an entry to wiki/log.md.

    Format: ``## [YYYY-MM-DD HH:MM:SS] operation | description``
    """
    log_path = wiki_dir / "log.md"
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"## [{date_str}] {operation} | {description}\n\n"

    kb_dir = wiki_dir.parent if wiki_dir.name == "wiki" else None
    with maybe_kb_ingest_lock(kb_dir, assume_locked=assume_locked):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not log_path.exists()
        with log_path.open("a", encoding="utf-8") as fh:
            if needs_header:
                fh.write("# Operations Log\n\n")
            fh.write(entry)
            fh.flush()
            os.fsync(fh.fileno())
