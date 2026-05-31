"""Tests for openkb.converter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


from openkb.converter import convert_document, get_pdf_page_count
from openkb.pdf_parser import MinerUParsedPdf


# ---------------------------------------------------------------------------
# get_pdf_page_count
# ---------------------------------------------------------------------------


class TestGetPdfPageCount:
    def test_returns_page_count(self, tmp_path):
        """Mock pypdf to return a doc with 5 pages."""
        fake_reader = MagicMock()
        fake_reader.pages = [object()] * 5
        with patch("openkb.pdf_parser.PdfReader", return_value=fake_reader):
            count = get_pdf_page_count(tmp_path / "fake.pdf")
        assert count == 5


# ---------------------------------------------------------------------------
# convert_document — .md input
# ---------------------------------------------------------------------------


class TestConvertDocumentMarkdown:
    def test_md_file_copied_to_wiki_sources(self, kb_dir):
        """A .md file is read and saved under wiki/sources/."""
        src = kb_dir / "raw" / "notes.md"
        src.write_text("# Notes\n\nSome content here.", encoding="utf-8")

        result = convert_document(src, kb_dir)

        assert result.skipped is False
        assert result.is_long_doc is False
        assert result.source_path is not None
        assert result.source_path.exists()
        assert result.source_path.read_text(encoding="utf-8").startswith("# Notes")

    def test_md_duplicate_skipped(self, kb_dir):
        """Second call with same file returns skipped=True when hash is registered."""
        from openkb.state import HashRegistry

        src = kb_dir / "raw" / "notes.md"
        src.write_text("# Notes\n\nSome content here.", encoding="utf-8")

        result1 = convert_document(src, kb_dir)  # first call
        # Simulate CLI registering the hash after successful compilation
        registry = HashRegistry(kb_dir / ".openkb" / "hashes.json")
        registry.add(result1.file_hash, {"name": src.name, "type": "md"})

        result2 = convert_document(src, kb_dir)  # second call
        assert result2.skipped is True
        assert result2.source_path is None
        assert result2.raw_path is None

    def test_md_raw_file_copied(self, kb_dir):
        """The original file should also be copied to raw/."""
        src = kb_dir / "input" / "notes.md"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("# Notes\n", encoding="utf-8")

        result = convert_document(src, kb_dir)

        assert result.raw_path is not None
        assert result.raw_path.exists()


# ---------------------------------------------------------------------------
# convert_document — PDF short doc
# ---------------------------------------------------------------------------


class TestConvertDocumentPdfShort:
    def test_short_pdf_converted_via_mineru(self, kb_dir, tmp_path):
        """PDF under threshold is converted with MinerU."""
        src = tmp_path / "short.pdf"
        src.write_bytes(b"%PDF-1.4 fake content")

        with (
            patch("openkb.converter.get_pdf_page_count", return_value=5),
            patch("openkb.converter.convert_pdf_to_markdown", return_value="# Short PDF\n\nConverted.") as mock_convert,
        ):
            result = convert_document(src, kb_dir)

        mock_convert.assert_called_once()
        assert result.skipped is False
        assert result.is_long_doc is False
        assert result.source_path is not None
        assert result.source_path.exists()

    def test_page_count_failure_uses_mineru_output_for_short_pdf(self, kb_dir, tmp_path):
        """If PDF metadata page count fails, MinerU output still drives conversion."""
        src = tmp_path / "short.pdf"
        src.write_bytes(b"%PDF-1.4 fake content")

        parsed = MinerUParsedPdf(
            markdown="# Short PDF\n\nConverted.",
            pages=[{"page": 1, "content": "Converted.", "images": []}],
        )

        with (
            patch("openkb.converter.get_pdf_page_count", side_effect=ValueError("bad pdf")),
            patch("openkb.converter.parse_pdf_with_mineru", return_value=parsed) as mock_parse,
        ):
            result = convert_document(src, kb_dir)

        mock_parse.assert_called_once()
        assert result.is_long_doc is False
        assert result.source_path is not None
        assert result.source_path.read_text(encoding="utf-8").startswith("# Short PDF")


# ---------------------------------------------------------------------------
# convert_document — PDF long doc
# ---------------------------------------------------------------------------


class TestConvertDocumentPdfLong:
    def test_long_pdf_returns_is_long_doc(self, kb_dir, tmp_path):
        """PDF >= threshold pages returns is_long_doc=True, source_path=None."""
        src = tmp_path / "long.pdf"
        src.write_bytes(b"%PDF-1.4 fake long content")

        with (
            patch("openkb.converter.get_pdf_page_count", return_value=200),
        ):
            result = convert_document(src, kb_dir)

        assert result.is_long_doc is True
        assert result.source_path is None
        assert result.skipped is False
        assert result.raw_path is not None

    def test_page_count_failure_uses_mineru_output_for_long_pdf(self, kb_dir, tmp_path):
        """MinerU page output can still route a PDF to PageIndex when page count fails."""
        src = tmp_path / "long.pdf"
        src.write_bytes(b"%PDF-1.4 fake content")

        parsed = MinerUParsedPdf(
            markdown="# Long PDF",
            pages=[{"page": page, "content": f"Page {page}", "images": []} for page in range(1, 21)],
        )

        with (
            patch("openkb.converter.get_pdf_page_count", side_effect=ValueError("bad pdf")),
            patch("openkb.converter.parse_pdf_with_mineru", return_value=parsed),
        ):
            result = convert_document(src, kb_dir)

        assert result.is_long_doc is True
        assert result.source_path is None
        assert result.raw_path is not None
