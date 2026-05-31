"""PDF parsing facade backed by MinerU."""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

logger = logging.getLogger(__name__)

_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((?!https?://|data:)([^)]+)\)")


class PdfParseError(RuntimeError):
    """Raised when MinerU cannot parse a PDF into OpenKB source content."""


@dataclass(frozen=True)
class MinerUParsedPdf:
    """Normalized MinerU output used by OpenKB."""

    markdown: str
    pages: list[dict[str, Any]]


def get_pdf_page_count(path: Path) -> int:
    """Return the number of pages in *path* without using PyMuPDF."""
    reader = PdfReader(str(path))
    return len(reader.pages)


def convert_pdf_to_markdown(
    pdf_path: Path,
    doc_name: str,
    images_dir: Path,
    output_root: Path,
    backend: str = "hybrid-auto-engine",
) -> str:
    """Convert a PDF to Markdown using MinerU."""
    parsed = parse_pdf_with_mineru(pdf_path, doc_name, images_dir, output_root, backend)
    return parsed.markdown


def convert_pdf_to_pages(
    pdf_path: Path,
    doc_name: str,
    images_dir: Path,
    output_root: Path,
    backend: str = "hybrid-auto-engine",
) -> list[dict[str, Any]]:
    """Convert a PDF to OpenKB's per-page JSON source format using MinerU."""
    parsed = parse_pdf_with_mineru(pdf_path, doc_name, images_dir, output_root, backend)
    if parsed.pages:
        return parsed.pages
    return _markdown_to_single_page(parsed.markdown)


def parse_pdf_with_mineru(
    pdf_path: Path,
    doc_name: str,
    images_dir: Path,
    output_root: Path,
    backend: str = "hybrid-auto-engine",
) -> MinerUParsedPdf:
    """Run MinerU and normalize its Markdown, JSON, and image outputs."""
    run_dir = output_root / doc_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    _run_mineru(pdf_path, run_dir, backend)
    doc_dir = _find_mineru_doc_dir(run_dir)

    markdown_path = _find_first(doc_dir, ["full.md", "*.md"])
    if markdown_path is None:
        raise PdfParseError(f"MinerU did not produce Markdown for {pdf_path.name}")

    markdown = markdown_path.read_text(encoding="utf-8")
    markdown = _copy_and_rewrite_markdown_images(markdown, markdown_path.parent, doc_name, images_dir)
    pages = _load_pages(doc_dir, doc_name, images_dir)
    return MinerUParsedPdf(markdown=markdown, pages=pages)


def _run_mineru(pdf_path: Path, output_dir: Path, backend: str) -> None:
    cmd = ["mineru", "-p", str(pdf_path), "-o", str(output_dir), "-b", backend]
    logger.info("Running MinerU: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise PdfParseError(
            "MinerU CLI not found. Install OpenKB dependencies with MinerU support."
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise PdfParseError(f"MinerU failed for {pdf_path.name}: {details}") from exc


def _find_mineru_doc_dir(run_dir: Path) -> Path:
    markdown = _find_first(run_dir, ["full.md", "*.md"])
    if markdown is not None:
        return markdown.parent
    content = _find_content_list(run_dir)
    if content is not None:
        return content.parent
    raise PdfParseError(f"MinerU output not found under {run_dir}")


def _find_first(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _find_content_list(root: Path) -> Path | None:
    for pattern in ("*content_list_v2.json", "*_content_list_v2.json", "*content_list.json", "*_content_list.json"):
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _load_pages(doc_dir: Path, doc_name: str, images_dir: Path) -> list[dict[str, Any]]:
    content_path = _find_content_list(doc_dir)
    if content_path is None:
        return []

    try:
        items = json.loads(content_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PdfParseError(f"MinerU content list is invalid JSON: {content_path}") from exc

    if not isinstance(items, list):
        return []

    by_page: dict[int, list[str]] = {}
    images_by_page: dict[int, list[dict[str, str]]] = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        page = int(item.get("page_idx", 0)) + 1
        text = _content_item_to_markdown(item, content_path.parent, doc_name, images_dir)
        if text:
            by_page.setdefault(page, []).append(text)
        if item.get("type") == "image" and item.get("img_path"):
            img_path = _copy_relative_asset(str(item["img_path"]), content_path.parent, doc_name, images_dir)
            if img_path:
                images_by_page.setdefault(page, []).append({"path": img_path})

    return [
        {
            "page": page,
            "content": "\n\n".join(parts),
            "images": images_by_page.get(page, []),
        }
        for page, parts in sorted(by_page.items())
    ]


def _content_item_to_markdown(
    item: dict[str, Any],
    source_dir: Path,
    doc_name: str,
    images_dir: Path,
) -> str:
    kind = item.get("type")
    if kind == "text":
        return str(item.get("text", "")).strip()
    if kind == "equation":
        return str(item.get("text", "")).strip()
    if kind == "table":
        body = str(item.get("table_body") or item.get("text") or "").strip()
        caption = _join_strings(item.get("table_caption"))
        return "\n\n".join(part for part in (caption, body) if part)
    if kind == "image":
        img_path = _copy_relative_asset(str(item.get("img_path", "")), source_dir, doc_name, images_dir)
        caption = _join_strings(item.get("image_caption"))
        if img_path:
            return "\n".join(part for part in (f"![image]({img_path})", caption) if part)
        return caption
    return str(item.get("text", "")).strip()


def _join_strings(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _copy_and_rewrite_markdown_images(
    markdown: str,
    source_dir: Path,
    doc_name: str,
    images_dir: Path,
) -> str:
    result = markdown
    for match in _MARKDOWN_IMAGE_RE.finditer(markdown):
        alt, rel_path = match.group(1), match.group(2)
        new_path = _copy_relative_asset(rel_path, source_dir, doc_name, images_dir)
        if not new_path:
            continue
        result = result.replace(match.group(0), f"![{alt}]({new_path})", 1)
    return result


def _copy_relative_asset(
    rel_path: str,
    source_dir: Path,
    doc_name: str,
    images_dir: Path,
) -> str | None:
    rel_path = rel_path.strip()
    if not rel_path:
        return None
    src = (source_dir / rel_path).resolve()
    if not src.exists() or not src.is_file():
        logger.warning("MinerU referenced missing image: %s", rel_path)
        return None

    images_dir.mkdir(parents=True, exist_ok=True)
    dest = images_dir / src.name
    counter = 2
    while dest.exists() and dest.read_bytes() != src.read_bytes():
        dest = images_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    if not dest.exists():
        shutil.copy2(src, dest)
    return f"sources/images/{doc_name}/{dest.name}"


def _markdown_to_single_page(markdown: str) -> list[dict[str, Any]]:
    return [{"page": 1, "content": markdown, "images": []}]
