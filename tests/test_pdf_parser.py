"""Tests for MinerU-backed PDF parsing."""
from __future__ import annotations

import json
from unittest.mock import patch

from openkb.pdf_parser import convert_pdf_to_markdown, convert_pdf_to_pages


def _seed_mineru_output(output_root, doc_name):
    doc_dir = output_root / doc_name / "mineru-result"
    images = doc_dir / "images"
    images.mkdir(parents=True)
    (images / "fig.jpg").write_bytes(b"fake-image")
    (doc_dir / "full.md").write_text(
        "# Paper\n\n![figure](images/fig.jpg)\n",
        encoding="utf-8",
    )
    (doc_dir / "sample_content_list.json").write_text(
        json.dumps(
            [
                {"type": "text", "text": "Intro text", "page_idx": 0},
                {
                    "type": "equation",
                    "text": "$$x=1$$",
                    "page_idx": 0,
                },
                {
                    "type": "image",
                    "img_path": "images/fig.jpg",
                    "image_caption": ["Figure caption."],
                    "page_idx": 1,
                },
                {
                    "type": "table",
                    "table_caption": ["Table caption."],
                    "table_body": "<table><tr><td>A</td></tr></table>",
                    "page_idx": 1,
                },
            ]
        ),
        encoding="utf-8",
    )


def test_convert_pdf_to_markdown_rewrites_mineru_images(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    output_root = tmp_path / "mineru"
    images_dir = tmp_path / "wiki" / "sources" / "images" / "sample"

    def fake_run(pdf_path, run_dir, backend):
        _seed_mineru_output(output_root, "sample")

    with patch("openkb.pdf_parser._run_mineru", side_effect=fake_run):
        markdown = convert_pdf_to_markdown(pdf, "sample", images_dir, output_root)

    assert "![figure](sources/images/sample/fig.jpg)" in markdown
    assert (images_dir / "fig.jpg").exists()


def test_convert_pdf_to_pages_groups_mineru_content_by_page(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    output_root = tmp_path / "mineru"
    images_dir = tmp_path / "wiki" / "sources" / "images" / "sample"

    def fake_run(pdf_path, run_dir, backend):
        _seed_mineru_output(output_root, "sample")

    with patch("openkb.pdf_parser._run_mineru", side_effect=fake_run):
        pages = convert_pdf_to_pages(pdf, "sample", images_dir, output_root)

    assert pages[0]["page"] == 1
    assert "Intro text" in pages[0]["content"]
    assert "$$x=1$$" in pages[0]["content"]
    assert pages[1]["page"] == 2
    assert "sources/images/sample/fig.jpg" in pages[1]["content"]
    assert "Table caption." in pages[1]["content"]
    assert pages[1]["images"] == [{"path": "sources/images/sample/fig.jpg"}]
