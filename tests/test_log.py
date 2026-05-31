from pathlib import Path
from unittest.mock import patch

from openkb.log import append_log


def test_append_log_appends_without_rewriting_existing_file(tmp_path, monkeypatch):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    log_path = wiki_dir / "log.md"
    log_path.write_text("# Log\n\n", encoding="utf-8")

    def fail_read_text(self: Path, *args, **kwargs):
        if self == log_path:
            raise AssertionError("append_log should not read the full log")
        return original_read_text(self, *args, **kwargs)

    original_read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", fail_read_text)

    with patch("openkb.log.os.fsync") as fsync:
        append_log(wiki_dir, "ingest", "paper.pdf", assume_locked=True)

    content = log_path.read_bytes().decode("utf-8")
    assert content.startswith("# Log\n\n")
    assert "ingest | paper.pdf" in content
    fsync.assert_called_once()


def test_append_log_creates_missing_log_with_header(tmp_path):
    wiki_dir = tmp_path / "wiki"

    with patch("openkb.log.os.fsync") as fsync:
        append_log(wiki_dir, "remove", "paper.pdf", assume_locked=True)

    content = (wiki_dir / "log.md").read_text(encoding="utf-8")
    assert content.startswith("# Operations Log\n\n")
    assert "remove | paper.pdf" in content
    fsync.assert_called_once()
