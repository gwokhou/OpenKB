from __future__ import annotations

import hashlib
import json
from pathlib import Path

from openkb.locks import atomic_write_json


class HashRegistry:
    """Persistent registry mapping file SHA-256 hashes to metadata dicts."""

    def __init__(self, path: Path) -> None:
        self._path = path
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                self._data: dict[str, dict] = json.load(fh)
        else:
            self._data = {}

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def is_known(self, file_hash: str) -> bool:
        """Return True if file_hash is already registered."""
        return file_hash in self._data

    def get(self, file_hash: str) -> dict | None:
        """Return metadata for file_hash, or None if not found."""
        return self._data.get(file_hash)

    def all_entries(self) -> dict[str, dict]:
        """Return a shallow copy of all hash -> metadata entries."""
        return dict(self._data)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, file_hash: str, metadata: dict) -> None:
        """Register file_hash with metadata and persist to disk."""
        new_data = dict(self._data)
        new_data[file_hash] = metadata
        self._persist(new_data)
        self._data = new_data

    def update(self, file_hash: str, updates: dict) -> bool:
        """Merge metadata updates for ``file_hash``. Returns True if updated."""
        if file_hash not in self._data:
            return False
        new_data = dict(self._data)
        new_metadata = dict(new_data[file_hash])
        new_metadata.update(updates)
        new_data[file_hash] = new_metadata
        self._persist(new_data)
        self._data = new_data
        return True

    def mark_pageindex_missing(self, file_hash: str, reason: str) -> bool:
        """Flag a long-PDF registry entry whose PageIndex document may be gone."""
        return self.update(
            file_hash,
            {
                "pageindex_missing": True,
                "pageindex_missing_reason": reason,
                "pageindex_uncertain": False,
            },
        )

    def mark_pageindex_uncertain(self, file_hash: str, reason: str) -> bool:
        """Flag a long-PDF registry entry whose PageIndex delete outcome is unknown."""
        return self.update(
            file_hash,
            {
                "pageindex_uncertain": True,
                "pageindex_uncertain_reason": reason,
                "pageindex_missing": False,
            },
        )

    def remove_by_doc_name(self, doc_name: str) -> bool:
        """Remove the entry whose metadata['doc_name'] matches. Returns True if removed."""
        new_data = dict(self._data)
        for file_hash, meta in list(new_data.items()):
            if meta.get("doc_name") == doc_name:
                del new_data[file_hash]
                self._persist(new_data)
                self._data = new_data
                return True
        return False

    def remove_by_hash(self, file_hash: str) -> bool:
        """Remove the entry keyed by ``file_hash``. Returns True if removed.

        Preferred over :meth:`remove_by_doc_name` when the caller already
        has the hash in hand — works regardless of whether the entry's
        metadata carries a ``doc_name`` field (legacy entries written
        before commit c504e26 do not).
        """
        if file_hash not in self._data:
            return False
        new_data = dict(self._data)
        del new_data[file_hash]
        self._persist(new_data)
        self._data = new_data
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _persist(self, data: dict[str, dict]) -> None:
        atomic_write_json(self._path, data)

    # ------------------------------------------------------------------
    # Static utility
    # ------------------------------------------------------------------

    @staticmethod
    def hash_file(path: Path) -> str:
        """Return the SHA-256 hex digest (64 chars) of the file at path."""
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
