"""Document conversion pipeline for OpenKB."""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from markitdown import MarkItDown

from openkb.config import load_config
from openkb.images import copy_relative_images, extract_base64_images
from openkb.locks import atomic_write_text, kb_ingest_lock
from openkb.pdf_parser import convert_pdf_to_markdown, get_pdf_page_count, parse_pdf_with_mineru
from openkb.state import HashRegistry

logger = logging.getLogger(__name__)


@dataclass
class ConvertResult:
    """Result returned by :func:`convert_document`."""

    raw_path: Path | None = None
    source_path: Path | None = None
    is_long_doc: bool = False
    skipped: bool = False
    file_hash: str | None = None  # For deferred hash registration
    staging_dir: Path | None = None


def convert_document(
    src: Path,
    kb_dir: Path,
    *,
    assume_locked: bool = False,
    staging_dir: Path | None = None,
) -> ConvertResult:
    """Convert a document and integrate it into the knowledge base.

    Steps:
    1. Hash-check — skip if already known.
    2. Copy source to ``raw/``.
    3. If PDF and page count >= threshold → return :attr:`ConvertResult.is_long_doc`.
    4. If ``.md`` — read, process relative images, save to ``wiki/sources/``.
    5. Otherwise — run MarkItDown, extract base64 images, save to ``wiki/sources/``.
    6. Register hash in the registry.
    """
    if not assume_locked:
        with kb_ingest_lock(kb_dir / ".openkb"):
            return convert_document(
                src,
                kb_dir,
                assume_locked=True,
                staging_dir=staging_dir,
            )

    # ------------------------------------------------------------------
    # Load config & state
    # ------------------------------------------------------------------
    openkb_dir = kb_dir / ".openkb"
    config = load_config(openkb_dir / "config.yaml")
    threshold: int = config.get("pageindex_threshold", 20)
    mineru_backend: str = config.get("mineru_backend", "hybrid-auto-engine")
    artifact_root = staging_dir if staging_dir is not None else kb_dir
    mineru_output_dir = artifact_root / config.get("mineru_output_dir", ".openkb/mineru")
    registry = HashRegistry(openkb_dir / "hashes.json")

    # ------------------------------------------------------------------
    # 1. Hash check
    # ------------------------------------------------------------------
    file_hash = HashRegistry.hash_file(src)
    if registry.is_known(file_hash):
        logger.info("Skipping already-known file: %s", src.name)
        return ConvertResult(skipped=True, staging_dir=staging_dir)

    # ------------------------------------------------------------------
    # 2. Copy to raw/
    # ------------------------------------------------------------------
    raw_dir = artifact_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_dest = raw_dir / src.name
    if raw_dest.resolve() != src.resolve():
        shutil.copy2(src, raw_dest)

    # ------------------------------------------------------------------
    # 3. PDF long-doc detection
    # ------------------------------------------------------------------
    pdf_page_count: int | None = None
    if src.suffix.lower() == ".pdf":
        try:
            pdf_page_count = get_pdf_page_count(src)
        except Exception as exc:
            logger.warning("Could not read PDF page count before MinerU parse for %s: %s", src.name, exc)
        if pdf_page_count is not None and pdf_page_count >= threshold:
            logger.info(
                "Long PDF detected (%d pages >= %d threshold): %s",
                pdf_page_count,
                threshold,
                src.name,
            )
            return ConvertResult(
                raw_path=raw_dest,
                is_long_doc=True,
                file_hash=file_hash,
                staging_dir=staging_dir,
            )

    # ------------------------------------------------------------------
    # 4/5. Convert to Markdown
    # ------------------------------------------------------------------
    sources_dir = artifact_root / "wiki" / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    images_dir = artifact_root / "wiki" / "sources" / "images" / src.stem
    images_dir.mkdir(parents=True, exist_ok=True)

    doc_name = src.stem

    if src.suffix.lower() == ".md":
        markdown = src.read_text(encoding="utf-8")
        markdown = copy_relative_images(
            markdown,
            src.parent,
            doc_name,
            images_dir,
            assume_locked=True,
        )
    elif src.suffix.lower() == ".pdf":
        if pdf_page_count is None:
            parsed = parse_pdf_with_mineru(
                src,
                doc_name,
                images_dir,
                mineru_output_dir,
                mineru_backend,
                assume_locked=True,
            )
            inferred_page_count = len(parsed.pages)
            if inferred_page_count >= threshold:
                logger.info(
                    "Long PDF detected from MinerU output (%d pages >= %d threshold): %s",
                    inferred_page_count,
                    threshold,
                    src.name,
                )
                return ConvertResult(
                    raw_path=raw_dest,
                    is_long_doc=True,
                    file_hash=file_hash,
                    staging_dir=staging_dir,
                )
            markdown = parsed.markdown
        else:
            markdown = convert_pdf_to_markdown(
                src,
                doc_name,
                images_dir,
                mineru_output_dir,
                mineru_backend,
                assume_locked=True,
            )
    else:
        # Non-PDF, non-MD: use markitdown (docx, pptx, html, etc.)
        mid = MarkItDown()
        result = mid.convert(str(src))
        markdown = result.text_content
        markdown = extract_base64_images(
            markdown,
            doc_name,
            images_dir,
            assume_locked=True,
        )

    dest_md = sources_dir / f"{doc_name}.md"
    atomic_write_text(dest_md, markdown)

    return ConvertResult(
        raw_path=raw_dest,
        source_path=dest_md,
        file_hash=file_hash,
        staging_dir=staging_dir,
    )
