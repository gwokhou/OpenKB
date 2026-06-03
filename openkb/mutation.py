"""Recoverable mutation snapshots for file-backed KB operations."""
from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from openkb.locks import atomic_write_binary, atomic_write_json, kb_ingest_lock


def _copy_file_atomic(src: Path, dest: Path) -> None:
    with src.open("rb") as source, atomic_write_binary(dest) as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


@dataclass
class MutationSnapshot:
    """Snapshot of final KB paths touched by a mutation attempt."""

    root: Path
    backup_dir: Path
    entries: dict[Path, Path | None] = field(default_factory=dict)
    journal_path: Path | None = None
    operation: str = "mutation"
    details: dict = field(default_factory=dict)

    def _journal_data(self, status: str) -> dict:
        return {
            "version": 1,
            "id": self.backup_dir.name.removeprefix("rollback-"),
            "operation": self.operation,
            "status": status,
            "kb_dir": str(self.root),
            "backup_dir": str(self.backup_dir),
            "details": self.details,
            "entries": [
                {
                    "target": str(target),
                    "backup": str(backup) if backup is not None else None,
                }
                for target, backup in self.entries.items()
            ],
        }

    def write_journal(self, status: str) -> None:
        if self.journal_path is not None:
            atomic_write_json(self.journal_path, self._journal_data(status))

    def rollback(self) -> None:
        for target, backup in sorted(
            self.entries.items(),
            key=lambda item: len(item[0].parts),
            reverse=True,
        ):
            self._restore_one(target, backup)
        self.write_journal("rolled_back")

    def rollback_best_effort(self) -> Exception | None:
        try:
            self.rollback()
        except Exception as exc:
            return exc
        return None

    def _restore_one(self, target: Path, backup: Path | None) -> None:
        if backup is None:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
            return

        if target.exists():
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        if backup.is_dir():
            shutil.copytree(backup, target)
        else:
            _copy_file_atomic(backup, target)

    def discard(self) -> None:
        self.write_journal("committed")
        shutil.rmtree(self.backup_dir, ignore_errors=True)
        if self.journal_path is not None:
            self.journal_path.unlink(missing_ok=True)

    def discard_best_effort(self) -> Exception | None:
        try:
            self.discard()
        except Exception as exc:
            return exc
        return None


def snapshot_paths(
    kb_dir: Path,
    paths: list[Path],
    *,
    operation: str = "mutation",
    details: dict | None = None,
) -> MutationSnapshot:
    """Snapshot final KB paths without acquiring a lock."""
    kb_dir = kb_dir.resolve()
    journal_id = uuid.uuid4().hex
    backup_dir = kb_dir / ".openkb" / "staging" / f"rollback-{journal_id}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    journal_path = kb_dir / ".openkb" / "journal" / f"{journal_id}.json"
    snapshot = MutationSnapshot(
        root=kb_dir,
        backup_dir=backup_dir,
        journal_path=journal_path,
        operation=operation,
        details=details or {},
    )

    for path in paths:
        target = path.resolve()
        if target in snapshot.entries:
            continue
        if not target.exists():
            snapshot.entries[target] = None
            continue
        rel = target.relative_to(kb_dir)
        backup = backup_dir / rel
        backup.parent.mkdir(parents=True, exist_ok=True)
        if target.is_dir():
            shutil.copytree(target, backup)
        else:
            shutil.copy2(target, backup)
        snapshot.entries[target] = backup

    snapshot.write_journal("active")
    return snapshot


def _snapshot_from_journal(path: Path, data: dict) -> MutationSnapshot:
    entries = {
        Path(item["target"]): Path(item["backup"]) if item.get("backup") else None
        for item in data.get("entries", [])
    }
    return MutationSnapshot(
        root=Path(data["kb_dir"]),
        backup_dir=Path(data["backup_dir"]),
        entries=entries,
        journal_path=path,
        operation=data.get("operation", "mutation"),
        details=data.get("details") or {},
    )


def recover_pending_journals(kb_dir: Path) -> list[str]:
    """Roll back active mutation journals left by interrupted processes."""
    kb_dir = kb_dir.resolve()
    journal_dir = kb_dir / ".openkb" / "journal"
    if not journal_dir.is_dir():
        return []

    messages: list[str] = []
    with kb_ingest_lock(kb_dir / ".openkb"):
        for journal_path in sorted(journal_dir.glob("*.json")):
            try:
                data = json.loads(journal_path.read_text(encoding="utf-8"))
                status = data.get("status", "active")
                snapshot = _snapshot_from_journal(journal_path, data)
                if status in {"active", "commit_started"}:
                    snapshot.rollback()
                    snapshot.discard()
                    messages.append(
                        f"Rolled back interrupted {snapshot.operation} journal {journal_path.name}."
                    )
                elif status in {"committed", "rolled_back"}:
                    snapshot.discard()
                    messages.append(f"Cleaned terminal mutation journal {journal_path.name}.")
                else:
                    messages.append(f"Left unknown mutation journal for manual inspection: {journal_path.name}.")
            except Exception as exc:
                messages.append(f"Could not recover journal {journal_path.name}: {type(exc).__name__}: {exc}")
    return messages
