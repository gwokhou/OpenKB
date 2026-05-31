from pathlib import Path

from openkb.mutation import MutationSnapshot, recover_pending_journals, snapshot_paths
from openkb.state import HashRegistry


def test_snapshot_rollback_best_effort_continues_after_one_failure(tmp_path, monkeypatch):
    keep = tmp_path / "keep.txt"
    fail = tmp_path / "fail.txt"
    keep.write_text("old")
    fail.write_text("old")

    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    keep_backup = backup_dir / "keep.txt"
    fail_backup = backup_dir / "fail.txt"
    keep_backup.write_text("old")
    fail_backup.write_text("old")

    snapshot = MutationSnapshot(
        root=tmp_path,
        backup_dir=backup_dir,
        entries={keep: keep_backup, fail: fail_backup},
    )
    keep.write_text("new")
    fail.write_text("new")

    original_restore = snapshot._restore_one

    def flaky_restore(target: Path, backup: Path | None) -> None:
        if target == fail:
            raise OSError("cannot restore")
        original_restore(target, backup)

    monkeypatch.setattr(snapshot, "_restore_one", flaky_restore)

    error = snapshot.rollback_best_effort()

    assert error is not None
    assert keep.read_text() == "old"
    assert fail.read_text() == "new"


def _seed_kb(tmp_path: Path) -> Path:
    kb = tmp_path
    (kb / ".openkb").mkdir()
    (kb / ".openkb" / "hashes.json").write_text("{}")
    return kb


def test_recover_active_journal_rolls_back_snapshot(tmp_path):
    kb = _seed_kb(tmp_path)
    target = kb / "wiki" / "index.md"
    target.parent.mkdir(parents=True)
    target.write_text("old")

    snapshot = snapshot_paths(
        kb,
        [target],
        operation="add",
        details={"file_hash": "h1", "doc_name": "doc"},
    )
    target.write_text("new")

    messages = recover_pending_journals(kb)

    assert target.read_text() == "old"
    assert any("Rolled back interrupted add journal" in msg for msg in messages)
    assert not snapshot.backup_dir.exists()
    assert list((kb / ".openkb" / "journal").glob("*.json")) == []


def test_recover_commit_started_add_finalizes_when_registry_committed(tmp_path):
    kb = _seed_kb(tmp_path)
    target = kb / "wiki" / "index.md"
    target.parent.mkdir(parents=True)
    target.write_text("old")

    snapshot = snapshot_paths(
        kb,
        [target],
        operation="add",
        details={"file_hash": "h1", "doc_name": "doc"},
    )
    target.write_text("new")
    snapshot.write_journal("commit_started")
    (kb / ".openkb" / "hashes.json").write_text('{"h1": {"name": "doc.md"}}')

    messages = recover_pending_journals(kb)

    assert target.read_text() == "new"
    assert any("Finalized committed add journal" in msg for msg in messages)
    assert not snapshot.backup_dir.exists()
    assert list((kb / ".openkb" / "journal").glob("*.json")) == []


def test_recover_commit_started_add_rolls_back_when_registry_missing(tmp_path):
    kb = _seed_kb(tmp_path)
    target = kb / "wiki" / "index.md"
    target.parent.mkdir(parents=True)
    target.write_text("old")

    snapshot = snapshot_paths(
        kb,
        [target],
        operation="add",
        details={"file_hash": "h1", "doc_name": "doc"},
    )
    target.write_text("new")
    snapshot.write_journal("commit_started")

    messages = recover_pending_journals(kb)

    assert target.read_text() == "old"
    assert any("Rolled back interrupted add journal" in msg for msg in messages)
    assert not snapshot.backup_dir.exists()
    assert list((kb / ".openkb" / "journal").glob("*.json")) == []


def test_recover_committed_remove_finishes_raw_cleanup(tmp_path):
    kb = _seed_kb(tmp_path)
    target = kb / "wiki" / "summaries" / "doc.md"
    raw = kb / "raw" / "doc.md"
    target.parent.mkdir(parents=True)
    raw.parent.mkdir(parents=True)
    target.write_text("old")
    raw.write_text("raw")

    snapshot = snapshot_paths(
        kb,
        [target, raw],
        operation="remove",
        details={"file_hash": "h1", "doc_name": "doc", "raw_path": str(raw)},
    )
    target.unlink()
    snapshot.write_journal("commit_started")

    messages = recover_pending_journals(kb)

    assert not raw.exists()
    assert not target.exists()
    assert any("Finalized committed remove journal" in msg for msg in messages)
    assert not snapshot.backup_dir.exists()
    assert list((kb / ".openkb" / "journal").glob("*.json")) == []


def test_recover_pageindex_deleted_remove_rolls_back_and_marks_registry(tmp_path):
    kb = _seed_kb(tmp_path)
    target = kb / "wiki" / "summaries" / "doc.md"
    target.parent.mkdir(parents=True)
    target.write_text("old")
    HashRegistry(kb / ".openkb" / "hashes.json").add(
        "h1",
        {
            "name": "doc.pdf",
            "doc_name": "doc",
            "type": "long_pdf",
            "doc_id": "pi-doc",
        },
    )

    snapshot = snapshot_paths(
        kb,
        [target],
        operation="remove",
        details={"file_hash": "h1", "doc_name": "doc", "doc_id": "pi-doc"},
    )
    target.unlink()
    snapshot.write_journal("pageindex_deleted")

    messages = recover_pending_journals(kb)

    assert target.read_text() == "old"
    metadata = HashRegistry(kb / ".openkb" / "hashes.json").get("h1")
    assert metadata is not None
    assert metadata["pageindex_missing"] is True
    assert metadata["pageindex_missing_reason"] == "pageindex_deleted"
    assert any("marked pageindex_missing" in msg for msg in messages)
    assert not snapshot.backup_dir.exists()
    assert list((kb / ".openkb" / "journal").glob("*.json")) == []


def test_recover_pageindex_started_remove_marks_uncertain_not_missing(tmp_path):
    kb = _seed_kb(tmp_path)
    target = kb / "wiki" / "summaries" / "doc.md"
    target.parent.mkdir(parents=True)
    target.write_text("old")
    HashRegistry(kb / ".openkb" / "hashes.json").add(
        "h1",
        {
            "name": "doc.pdf",
            "doc_name": "doc",
            "type": "long_pdf",
            "doc_id": "pi-doc",
        },
    )

    snapshot = snapshot_paths(
        kb,
        [target],
        operation="remove",
        details={"file_hash": "h1", "doc_name": "doc", "doc_id": "pi-doc"},
    )
    target.unlink()
    snapshot.write_journal("pageindex_delete_started")

    messages = recover_pending_journals(kb)

    assert target.read_text() == "old"
    metadata = HashRegistry(kb / ".openkb" / "hashes.json").get("h1")
    assert metadata is not None
    assert metadata["pageindex_uncertain"] is True
    assert metadata["pageindex_uncertain_reason"] == "pageindex_delete_started"
    assert metadata["pageindex_missing"] is False
    assert any("marked pageindex_uncertain" in msg for msg in messages)
    assert not snapshot.backup_dir.exists()
    assert list((kb / ".openkb" / "journal").glob("*.json")) == []


def test_recover_pageindex_deleted_remove_finalizes_when_registry_missing(tmp_path):
    kb = _seed_kb(tmp_path)
    target = kb / "wiki" / "summaries" / "doc.md"
    raw = kb / "raw" / "doc.pdf"
    target.parent.mkdir(parents=True)
    raw.parent.mkdir(parents=True)
    target.write_text("old")
    raw.write_text("raw")

    snapshot = snapshot_paths(
        kb,
        [target, raw],
        operation="remove",
        details={
            "file_hash": "h1",
            "doc_name": "doc",
            "doc_id": "pi-doc",
            "raw_path": str(raw),
        },
    )
    target.unlink()
    snapshot.write_journal("pageindex_deleted")

    messages = recover_pending_journals(kb)

    assert not raw.exists()
    assert not target.exists()
    assert any("Finalized committed remove journal" in msg for msg in messages)
    assert not snapshot.backup_dir.exists()
    assert list((kb / ".openkb" / "journal").glob("*.json")) == []


def test_recover_unknown_journal_status_keeps_journal_for_inspection(tmp_path):
    kb = _seed_kb(tmp_path)
    target = kb / "wiki" / "index.md"
    target.parent.mkdir(parents=True)
    target.write_text("old")

    snapshot = snapshot_paths(
        kb,
        [target],
        operation="add",
        details={"file_hash": "h1", "doc_name": "doc"},
    )
    target.write_text("new")
    snapshot.write_journal("future_status")

    messages = recover_pending_journals(kb)

    assert target.read_text() == "new"
    assert any("manual inspection" in msg for msg in messages)
    assert snapshot.backup_dir.exists()
    assert snapshot.journal_path is not None
    assert snapshot.journal_path.exists()


def test_recover_terminal_journal_status_cleans_backup(tmp_path):
    kb = _seed_kb(tmp_path)
    target = kb / "wiki" / "index.md"
    target.parent.mkdir(parents=True)
    target.write_text("old")

    snapshot = snapshot_paths(
        kb,
        [target],
        operation="add",
        details={"file_hash": "h1", "doc_name": "doc"},
    )
    snapshot.write_journal("rolled_back")

    messages = recover_pending_journals(kb)

    assert any("Cleaned terminal mutation journal" in msg for msg in messages)
    assert not snapshot.backup_dir.exists()
    assert snapshot.journal_path is not None
    assert not snapshot.journal_path.exists()
