import json

from click.testing import CliRunner

from openkb.cli import cli


def test_doctor_reports_pageindex_missing(kb_dir):
    (kb_dir / ".openkb" / "hashes.json").write_text(json.dumps({
        "h1": {
            "name": "doc.pdf",
            "doc_name": "doc",
            "type": "long_pdf",
            "pageindex_missing": True,
            "pageindex_missing_reason": "pageindex_deleted",
        },
    }))
    (kb_dir / "wiki" / "summaries" / "doc.md").write_text("# Doc\n")
    (kb_dir / "wiki" / "sources" / "doc.json").write_text("[]")

    result = CliRunner().invoke(cli, ["--kb-dir", str(kb_dir), "doctor"])

    assert result.exit_code == 0, result.output
    assert "PAGEINDEX_MISSING h1 doc.pdf reason=pageindex_deleted" in result.output


def test_doctor_reports_pageindex_uncertain(kb_dir):
    (kb_dir / ".openkb" / "hashes.json").write_text(json.dumps({
        "h1": {
            "name": "doc.pdf",
            "doc_name": "doc",
            "type": "long_pdf",
            "pageindex_uncertain": True,
            "pageindex_uncertain_reason": "pageindex_delete_started",
        },
    }))
    (kb_dir / "raw" / "doc.pdf").write_bytes(b"%PDF")
    (kb_dir / "wiki" / "summaries" / "doc.md").write_text("# Doc\n")
    (kb_dir / "wiki" / "sources" / "doc.json").write_text("[]")

    result = CliRunner().invoke(cli, ["--kb-dir", str(kb_dir), "doctor"])

    assert result.exit_code == 0, result.output
    assert "PAGEINDEX_UNCERTAIN h1 doc.pdf reason=pageindex_delete_started" in result.output


def test_doctor_reports_missing_raw(kb_dir):
    (kb_dir / ".openkb" / "hashes.json").write_text(json.dumps({
        "h1": {
            "name": "doc.md",
            "doc_name": "doc",
            "type": "markdown",
            "path": "raw/doc.md",
        },
    }))
    (kb_dir / "wiki" / "summaries" / "doc.md").write_text("# Doc\n")
    (kb_dir / "wiki" / "sources" / "doc.md").write_text("# Source\n")

    result = CliRunner().invoke(cli, ["--kb-dir", str(kb_dir), "doctor"])

    assert result.exit_code == 0, result.output
    assert "MISSING_RAW h1 raw/doc.md" in result.output


def test_doctor_repair_removes_orphan_rollback_staging(kb_dir):
    orphan = kb_dir / ".openkb" / "staging" / "rollback-orphan"
    orphan.mkdir(parents=True)
    (orphan / "file").write_text("backup")

    result = CliRunner().invoke(cli, ["--kb-dir", str(kb_dir), "doctor", "--repair"])

    assert result.exit_code == 0, result.output
    assert "Repaired: Removed orphan staging .openkb/staging/rollback-orphan" in result.output
    assert not orphan.exists()
