"""Tests for OpenKB KB locks and atomic writes."""
from __future__ import annotations

import json

import pytest

from openkb.locks import (
    atomic_write_json,
    atomic_write_text,
    kb_ingest_lock,
    kb_read_lock,
)


def test_write_lock_is_reentrant(tmp_path):
    openkb_dir = tmp_path / ".openkb"

    with kb_ingest_lock(openkb_dir):
        with kb_ingest_lock(openkb_dir):
            assert (openkb_dir / "ingest.lock").exists()


def test_read_lock_is_reentrant(tmp_path):
    openkb_dir = tmp_path / ".openkb"

    with kb_read_lock(openkb_dir):
        with kb_read_lock(openkb_dir):
            assert (openkb_dir / "ingest.lock").exists()


def test_read_to_write_upgrade_fails(tmp_path):
    openkb_dir = tmp_path / ".openkb"

    with kb_read_lock(openkb_dir):
        with pytest.raises(RuntimeError, match="Cannot upgrade"):
            with kb_ingest_lock(openkb_dir):
                pass


def test_write_lock_can_take_nested_read(tmp_path):
    openkb_dir = tmp_path / ".openkb"

    with kb_ingest_lock(openkb_dir):
        with kb_read_lock(openkb_dir):
            assert (openkb_dir / "ingest.lock").exists()


def test_atomic_write_text_replaces_file(tmp_path):
    target = tmp_path / "nested" / "file.txt"
    atomic_write_text(target, "first")
    atomic_write_text(target, "second")

    assert target.read_text(encoding="utf-8") == "second"
    assert list(target.parent.glob("*.tmp")) == []


def test_atomic_write_json_replaces_file(tmp_path):
    target = tmp_path / "hashes.json"

    atomic_write_json(target, {"a": {"name": "doc.pdf"}}, ensure_ascii=False)

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": {"name": "doc.pdf"}}
