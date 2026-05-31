"""Cooperative filesystem locks used by OpenKB mutation paths.

The lock is an advisory local-filesystem protocol for OpenKB processes that use
these helpers. It does not guarantee cross-host safety on network/synced
filesystems where ``fcntl.flock`` semantics may be unavailable or inconsistent.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import Iterator

_LOCKS_GUARD = threading.Lock()
_LOCAL_LOCKS: dict[Path, "_LocalRwLock"] = {}
_HELD_LOCKS = threading.local()


class _LocalRwLock:
    """In-process read/write gate for a filesystem lock path."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer = False

    @contextlib.contextmanager
    def hold(self, *, exclusive: bool) -> Iterator[None]:
        with self._condition:
            if exclusive:
                while self._writer or self._readers:
                    self._condition.wait()
                self._writer = True
            else:
                while self._writer:
                    self._condition.wait()
                self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                if exclusive:
                    self._writer = False
                else:
                    self._readers -= 1
                self._condition.notify_all()


def _held_locks() -> dict[Path, tuple[int, int]]:
    held = getattr(_HELD_LOCKS, "counts", None)
    if held is None:
        held = {}
        _HELD_LOCKS.counts = held
    return held


def _local_lock(lock_path: Path) -> _LocalRwLock:
    resolved = lock_path.resolve()
    with _LOCKS_GUARD:
        lock = _LOCAL_LOCKS.get(resolved)
        if lock is None:
            lock = _LocalRwLock()
            _LOCAL_LOCKS[resolved] = lock
        return lock


@contextlib.contextmanager
def kb_lock(openkb_dir: Path, *, exclusive: bool):
    """Hold a KB-level advisory lock.

    All CLI and library write paths use ``exclusive=True``. Read-only snapshot
    paths use ``exclusive=False`` so they can run together while still blocking
    during mutations.
    """
    lock_path = openkb_dir / "ingest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = lock_path.resolve()
    held = _held_locks()
    exclusive_depth, shared_depth = held.get(resolved, (0, 0))
    if exclusive_depth or shared_depth:
        if exclusive and not exclusive_depth:
            raise RuntimeError("Cannot upgrade an existing KB read lock to a write lock")
        held[resolved] = (
            exclusive_depth + (1 if exclusive else 0),
            shared_depth + (0 if exclusive else 1),
        )
        try:
            yield
        finally:
            current_exclusive, current_shared = held[resolved]
            next_counts = (
                current_exclusive - (1 if exclusive else 0),
                current_shared - (0 if exclusive else 1),
            )
            if next_counts == (0, 0):
                del held[resolved]
            else:
                held[resolved] = next_counts
        return

    with _local_lock(lock_path).hold(exclusive=exclusive):
        with lock_path.open("a+", encoding="utf-8") as fh:
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(fh.fileno(), mode)
            held[resolved] = (1, 0) if exclusive else (0, 1)
            try:
                yield
            finally:
                held.pop(resolved, None)
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def kb_ingest_lock(openkb_dir: Path):
    """Hold an exclusive KB lock."""
    return kb_lock(openkb_dir, exclusive=True)


def kb_read_lock(openkb_dir: Path):
    """Hold a shared KB lock."""
    return kb_lock(openkb_dir, exclusive=False)


def _initialized_kb_dir(kb_dir: Path | None) -> Path | None:
    """Return *kb_dir* only when it is already an initialized KB root."""
    if kb_dir is None or not (kb_dir / ".openkb").is_dir():
        return None
    return kb_dir


@contextlib.contextmanager
def maybe_kb_read_lock(kb_dir: Path | None, *, assume_locked: bool = False) -> Iterator[None]:
    """Take a read lock for an already-resolved initialized KB root.

    This helper deliberately does not discover the current/global KB. Callers
    must pass the KB root they derived from their own path boundary.
    """
    locked_kb_dir = _initialized_kb_dir(kb_dir)
    if assume_locked or locked_kb_dir is None:
        yield
        return
    with kb_read_lock(locked_kb_dir / ".openkb"):
        yield


@contextlib.contextmanager
def maybe_kb_ingest_lock(kb_dir: Path | None, *, assume_locked: bool = False) -> Iterator[None]:
    """Take a write lock for an already-resolved initialized KB root.

    This helper deliberately does not discover the current/global KB. Callers
    must pass the KB root they derived from their own path boundary.
    """
    locked_kb_dir = _initialized_kb_dir(kb_dir)
    if assume_locked or locked_kb_dir is None:
        yield
        return
    with kb_ingest_lock(locked_kb_dir / ".openkb"):
        yield


def _default_file_mode() -> int:
    current_umask = os.umask(0)
    os.umask(current_umask)
    return 0o666 & ~current_umask


def _target_mode(path: Path) -> int:
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        return _default_file_mode()


def _fsync_directory(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically replace a binary file with *content* and durable metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, _target_mode(path))
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    finally:
        tmp_path.unlink(missing_ok=True)


@contextlib.contextmanager
def atomic_write_binary(path: Path):
    """Open *path* for streamed atomic binary replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, _target_mode(path))
        with os.fdopen(fd, "wb") as fh:
            yield fh
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace a text file with *content* and durable metadata."""
    atomic_write_bytes(path, content.encode(encoding))


def atomic_write_json(
    path: Path,
    data: object,
    *,
    ensure_ascii: bool = True,
    default=None,
) -> None:
    """Atomically replace a JSON file."""
    atomic_write_text(
        path,
        json.dumps(data, indent=2, ensure_ascii=ensure_ascii, default=default) + "\n",
    )
