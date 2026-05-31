"""Transactional helpers for KB mutation paths."""
from __future__ import annotations

import logging
import shutil
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path

from openkb.locks import atomic_write_binary, atomic_write_json, kb_ingest_lock
from openkb.state import HashRegistry

logger = logging.getLogger(__name__)

_REMOVE_PAGEINDEX_STATUSES = {"pageindex_delete_started", "pageindex_deleted"}


def _copy_file_atomic(src: Path, dest: Path) -> None:
    with src.open("rb") as source, atomic_write_binary(dest) as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def _cleanup_pageindex_best_effort(kb_dir: Path, doc_name: str | None, doc_id: str | None) -> str | None:
    if not doc_id or not (kb_dir / ".openkb" / "pageindex.db").exists():
        return None
    try:
        from pageindex import PageIndexClient

        from openkb.config import DEFAULT_CONFIG, load_config

        config = load_config(kb_dir / ".openkb" / "config.yaml")
        model = config.get("model", DEFAULT_CONFIG["model"])
        client = PageIndexClient(model=model, storage_path=str(kb_dir / ".openkb"))
        client.collection().delete_document(doc_id)
        return f"Cleaned PageIndex doc {doc_id[:12]} for interrupted {doc_name or 'document'}."
    except Exception as exc:
        return f"Could not clean PageIndex doc {doc_id[:12]}: {type(exc).__name__}: {exc}"


@dataclass
class MutationSnapshot:
    """Snapshot of final KB paths touched by a commit attempt."""

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
        if self.journal_path is None:
            return
        atomic_write_json(self.journal_path, self._journal_data(status))

    def rollback(self) -> None:
        errors: list[str] = []
        for target, backup in sorted(
            self.entries.items(),
            key=lambda item: len(item[0].parts),
            reverse=True,
        ):
            try:
                self._restore_one(target, backup)
            except Exception as exc:
                errors.append(f"{target}: {type(exc).__name__}: {exc}")
        if errors:
            raise RuntimeError("Rollback failed for: " + "; ".join(errors))
        self.write_journal("rolled_back")

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

    def rollback_best_effort(self) -> Exception | None:
        try:
            self.rollback()
        except Exception as exc:
            return exc
        return None

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
    """Snapshot final KB paths without acquiring a mutation lock."""
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
    """Recover or finalize mutation journals left by a crashed process."""
    kb_dir = kb_dir.resolve()
    journal_dir = kb_dir / ".openkb" / "journal"
    if not journal_dir.is_dir():
        return []

    messages: list[str] = []
    registry = HashRegistry(kb_dir / ".openkb" / "hashes.json")
    for journal_path in sorted(journal_dir.glob("*.json")):
        try:
            import json

            data = json.loads(journal_path.read_text(encoding="utf-8"))
            status = data.get("status", "active")
            snapshot = _snapshot_from_journal(journal_path, data)
            operation = data.get("operation")
            details = data.get("details") or {}
            file_hash = details.get("file_hash")

            if operation == "remove" and status in _REMOVE_PAGEINDEX_STATUSES:
                registry_has_hash = bool(file_hash and registry.is_known(file_hash))
                if registry_has_hash:
                    if file_hash:
                        if status == "pageindex_deleted":
                            registry.mark_pageindex_missing(file_hash, status)
                            marker = "pageindex_missing"
                        else:
                            registry.mark_pageindex_uncertain(file_hash, status)
                            marker = "pageindex_uncertain"
                    snapshot.rollback()
                    snapshot.discard()
                    messages.append(
                        f"Rolled back interrupted remove journal {journal_path.name}; "
                        f"registry entry was marked {marker}."
                    )
                else:
                    raw_path = details.get("raw_path")
                    if raw_path:
                        Path(raw_path).unlink(missing_ok=True)
                    snapshot.discard()
                    messages.append(f"Finalized committed remove journal {journal_path.name}.")
                continue

            if status == "active":
                if operation == "add":
                    cleanup_msg = _cleanup_pageindex_best_effort(
                        kb_dir,
                        details.get("doc_name"),
                        details.get("doc_id"),
                    )
                    if cleanup_msg:
                        messages.append(cleanup_msg)
                snapshot.rollback()
                snapshot.discard()
                messages.append(f"Rolled back interrupted {operation or 'mutation'} journal {journal_path.name}.")
                continue

            if status in {"committed", "rolled_back"}:
                snapshot.discard()
                messages.append(f"Cleaned terminal mutation journal {journal_path.name}.")
                continue

            if status != "commit_started":
                messages.append(f"Left unknown mutation journal for manual inspection: {journal_path.name}.")
                continue

            registry_has_hash = bool(file_hash and registry.is_known(file_hash))
            if operation == "add":
                if registry_has_hash:
                    snapshot.discard()
                    messages.append(f"Finalized committed add journal {journal_path.name}.")
                else:
                    cleanup_msg = _cleanup_pageindex_best_effort(
                        kb_dir,
                        details.get("doc_name"),
                        details.get("doc_id"),
                    )
                    if cleanup_msg:
                        messages.append(cleanup_msg)
                    snapshot.rollback()
                    snapshot.discard()
                    messages.append(f"Rolled back interrupted add journal {journal_path.name}.")
            elif operation == "remove":
                if registry_has_hash:
                    if file_hash and details.get("pageindex_deleted"):
                        registry.mark_pageindex_missing(file_hash, "commit_started_after_pageindex_deleted")
                    snapshot.rollback()
                    snapshot.discard()
                    if details.get("pageindex_deleted"):
                        messages.append(
                            f"Rolled back interrupted remove journal {journal_path.name}; "
                            "registry entry was marked pageindex_missing."
                        )
                    else:
                        messages.append(f"Rolled back interrupted remove journal {journal_path.name}.")
                else:
                    raw_path = details.get("raw_path")
                    if raw_path:
                        Path(raw_path).unlink(missing_ok=True)
                    snapshot.discard()
                    messages.append(f"Finalized committed remove journal {journal_path.name}.")
            else:
                messages.append(f"Left unknown mutation journal for manual inspection: {journal_path.name}.")
        except Exception as exc:
            messages.append(f"Could not recover journal {journal_path.name}: {type(exc).__name__}: {exc}")
    return messages


@dataclass
class KbMutationContext:
    """Hold shared state for one serialized KB mutation."""

    kb_dir: Path
    _lock: AbstractContextManager[None] | None = field(init=False, default=None)
    _staging_dirs: list[Path] = field(init=False, default_factory=list)
    registry: HashRegistry = field(init=False)

    def __enter__(self) -> "KbMutationContext":
        self.kb_dir = self.kb_dir.resolve()
        self._lock = kb_ingest_lock(self.kb_dir / ".openkb")
        self._lock.__enter__()
        for message in recover_pending_journals(self.kb_dir):
            logger.warning(message)
        self.registry = HashRegistry(self.kb_dir / ".openkb" / "hashes.json")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for path in list(self._staging_dirs):
            self.cleanup_staging(path)
        if self._lock is not None:
            self._lock.__exit__(exc_type, exc, tb)

    def staging_dir(self, doc_name: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in doc_name)
        path = self.kb_dir / ".openkb" / "staging" / f"{safe}-{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=False)
        self._staging_dirs.append(path)
        return path

    def cleanup_staging(self, path: Path | None) -> None:
        if path is not None:
            shutil.rmtree(path, ignore_errors=True)
            if path in self._staging_dirs:
                self._staging_dirs.remove(path)

    def snapshot_paths(
        self,
        paths: list[Path],
        *,
        operation: str = "mutation",
        details: dict | None = None,
    ) -> MutationSnapshot:
        return snapshot_paths(
            self.kb_dir,
            paths,
            operation=operation,
            details=details,
        )

    def install_staged_tree(self, staging_dir: Path | None) -> None:
        """Copy staged raw/source artifacts into their final KB locations."""
        if staging_dir is None or not staging_dir.exists():
            return
        for rel in ("raw", "wiki/sources"):
            src = staging_dir / rel
            if not src.exists():
                continue
            dest = self.kb_dir / rel
            dest.mkdir(parents=True, exist_ok=True)
            for child in src.iterdir():
                target = dest / child.name
                if child.is_dir():
                    for nested in child.rglob("*"):
                        if nested.is_file():
                            _copy_file_atomic(nested, target / nested.relative_to(child))
                else:
                    _copy_file_atomic(child, target)
