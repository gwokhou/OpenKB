from pathlib import Path
import stat
import threading
import time

import pytest

from openkb.locks import atomic_write_text, kb_ingest_lock, kb_read_lock, maybe_kb_read_lock


def _openkb_dir(tmp_path: Path) -> Path:
    openkb_dir = tmp_path / ".openkb"
    openkb_dir.mkdir()
    return openkb_dir


def test_read_lock_can_nest_inside_write_lock(tmp_path):
    openkb_dir = _openkb_dir(tmp_path)

    with kb_ingest_lock(openkb_dir):
        with maybe_kb_read_lock(tmp_path):
            assert True


def test_write_lock_cannot_upgrade_from_read_lock(tmp_path):
    openkb_dir = _openkb_dir(tmp_path)

    with kb_read_lock(openkb_dir):
        with pytest.raises(RuntimeError, match="Cannot upgrade"):
            with kb_ingest_lock(openkb_dir):
                pass


def test_read_locks_are_shared_within_process(tmp_path):
    openkb_dir = _openkb_dir(tmp_path)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first_reader():
        with kb_read_lock(openkb_dir):
            first_entered.set()
            release_first.wait(timeout=2)

    t = threading.Thread(target=first_reader)
    t.start()
    assert first_entered.wait(timeout=2)

    with kb_read_lock(openkb_dir):
        second_entered.set()

    release_first.set()
    t.join(timeout=2)

    assert second_entered.is_set()
    assert not t.is_alive()


def test_write_lock_waits_for_in_process_readers(tmp_path):
    openkb_dir = _openkb_dir(tmp_path)
    first_entered = threading.Event()
    release_first = threading.Event()
    writer_entered = threading.Event()

    def first_reader():
        with kb_read_lock(openkb_dir):
            first_entered.set()
            release_first.wait(timeout=2)

    def writer():
        with kb_ingest_lock(openkb_dir):
            writer_entered.set()

    reader = threading.Thread(target=first_reader)
    reader.start()
    assert first_entered.wait(timeout=2)

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    time.sleep(0.05)
    assert not writer_entered.is_set()

    release_first.set()
    reader.join(timeout=2)
    writer_thread.join(timeout=2)

    assert writer_entered.is_set()
    assert not reader.is_alive()
    assert not writer_thread.is_alive()


def test_atomic_write_text_preserves_existing_mode(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o644)

    atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
