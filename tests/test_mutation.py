"""Tests for recoverable mutation snapshots."""
from __future__ import annotations

from openkb.mutation import recover_pending_journals, snapshot_paths


def test_snapshot_rolls_back_modified_and_created_paths(tmp_path):
    kb_dir = tmp_path
    (kb_dir / ".openkb").mkdir()
    existing = kb_dir / "wiki" / "sources" / "doc.md"
    created = kb_dir / "wiki" / "summaries" / "doc.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("before", encoding="utf-8")

    snapshot = snapshot_paths(kb_dir, [existing, created], operation="test")
    existing.write_text("after", encoding="utf-8")
    created.parent.mkdir(parents=True, exist_ok=True)
    created.write_text("new", encoding="utf-8")

    snapshot.rollback()

    assert existing.read_text(encoding="utf-8") == "before"
    assert not created.exists()


def test_recover_pending_journals_rolls_back_active_snapshot(tmp_path):
    kb_dir = tmp_path
    (kb_dir / ".openkb").mkdir()
    target = kb_dir / "wiki" / "sources" / "doc.md"
    target.parent.mkdir(parents=True)
    target.write_text("before", encoding="utf-8")

    snapshot_paths(kb_dir, [target], operation="test")
    target.write_text("after", encoding="utf-8")

    messages = recover_pending_journals(kb_dir)

    assert any("Rolled back interrupted test journal" in msg for msg in messages)
    assert target.read_text(encoding="utf-8") == "before"
    assert not list((kb_dir / ".openkb" / "journal").glob("*.json"))
