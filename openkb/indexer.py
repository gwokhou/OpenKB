"""PageIndex indexer for long documents."""
from __future__ import annotations

import hashlib
import json as json_mod
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pageindex import IndexConfig, PageIndexClient

from openkb.config import load_config
from openkb.tree_renderer import render_summary_md

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Result of indexing a long document via PageIndex."""

    doc_id: str
    description: str
    tree: dict


def _normalize_page_content(raw_pages: Any) -> list[dict[str, Any]]:
    """Normalize PageIndex/local PDF page content into OpenKB's JSON shape."""
    if not isinstance(raw_pages, list):
        return []

    pages: list[dict[str, Any]] = []
    for index, item in enumerate(raw_pages, start=1):
        if isinstance(item, str):
            content = item.strip()
            if content:
                pages.append({"page": index, "content": content, "images": []})
            continue

        if not isinstance(item, dict):
            continue

        raw_page = item.get("page", item.get("page_number", item.get("page_num", index)))
        try:
            page_number = int(raw_page)
        except (TypeError, ValueError):
            page_number = index
        if page_number < 1:
            page_number = index

        content = item.get("content", item.get("markdown", item.get("text", "")))
        if content is None:
            content = ""
        content = str(content).strip()

        images = item.get("images", [])
        if not isinstance(images, list):
            images = []
        normalized_images = [
            image for image in images
            if isinstance(image, dict) and isinstance(image.get("path"), str)
        ]

        if content or normalized_images:
            pages.append({
                "page": page_number,
                "content": content,
                "images": normalized_images,
            })

    return pages


def _uses_deepseek_json_mode(model: str | None) -> bool:
    if not model:
        return False
    normalized = model.removeprefix("litellm/")
    return normalized.startswith("deepseek/")


def _response_content(response) -> str:
    choice = response.choices[0]
    message = choice["message"] if isinstance(choice, dict) else choice.message
    if isinstance(message, dict):
        return message.get("content") or ""
    return message.content or ""


def _response_finish_reason(response) -> str | None:
    choice = response.choices[0]
    if isinstance(choice, dict):
        return choice.get("finish_reason")
    return getattr(choice, "finish_reason", None)


def _response_usage(response) -> str:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if not usage:
        return "usage=unknown"
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", "?")
        completion_tokens = usage.get("completion_tokens", "?")
        total_tokens = usage.get("total_tokens", "?")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", "?")
        completion_tokens = getattr(usage, "completion_tokens", "?")
        total_tokens = getattr(usage, "total_tokens", "?")
    return f"usage=in:{prompt_tokens},out:{completion_tokens},total:{total_tokens}"


def _prompt_diagnostics(prompt: str) -> tuple[str, int, str]:
    normalized = " ".join(prompt.split())
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    preview = normalized[:240]
    if len(normalized) > len(preview):
        preview += "..."
    return digest, len(prompt), preview


def _page_refs(prompt: str) -> str:
    refs = sorted({int(match) for match in re.findall(r"<physical_index_(\d+)>", prompt)})
    if not refs:
        return "none"
    if len(refs) == 1:
        return str(refs[0])
    return f"{refs[0]}-{refs[-1]} ({len(refs)} refs)"


def _prompt_kind(prompt: str) -> str:
    markers = [
        ("toc_detector_single_page", "detect if there is a table of content provided"),
        ("check_toc_extraction_complete", "check if the  table of contents is complete"),
        ("check_toc_transformation_complete", "check if the cleaned table of contents is complete"),
        ("extract_toc_content", "extract the full table of contents"),
        ("detect_page_index", "detect if there are page numbers/indices given"),
        ("toc_index_extractor", "add the physical_index to the table of contents"),
        ("toc_transformer", "transform the whole table of content into a JSON format"),
        ("add_page_number_to_toc", "The given structure contains the result of the previous part"),
        ("generate_toc_continue", "continue the tree structure from the previous part"),
        ("generate_toc_init", "generate the tree structure of the document"),
        ("check_title_appearance", "check if the given section appears or starts"),
        ("check_title_appearance_in_start", "check if the current section starts in the beginning"),
        ("single_toc_item_index_fixer", "find the physical index of the start page of the section"),
    ]
    for kind, marker in markers:
        if marker in prompt:
            return kind
    return "unknown"


def _patch_pageindex_deepseek_json_mode(model: str | None):
    """Patch PageIndex's local JSON prompts to use DeepSeek JSON Output mode."""
    if not _uses_deepseek_json_mode(model):
        return lambda: None

    import litellm
    import pageindex.index.page_index as page_index

    original_llm_completion = page_index.llm_completion
    original_llm_acompletion = page_index.llm_acompletion
    original_toc_detector = page_index.toc_detector_single_page

    def log_empty_response(
        *,
        mode: str,
        attempt: int,
        max_retries: int,
        model: str | None,
        prompt: str,
        response,
        chat_history_len: int = 0,
    ) -> None:
        prompt_hash, prompt_chars, prompt_preview = _prompt_diagnostics(prompt)
        finish_reason = _response_finish_reason(response)
        usage = _response_usage(response)
        kind = _prompt_kind(prompt)
        pages = _page_refs(prompt)
        logger.warning(
            "PageIndex returned empty DeepSeek JSON response "
            "(%d/%d, kind=%s, pages=%s, finish_reason=%s, prompt_hash=%s, %s)",
            attempt,
            max_retries,
            kind,
            pages,
            finish_reason,
            prompt_hash,
            usage,
        )
        logger.debug(
            "PageIndex empty DeepSeek JSON details "
            "(mode=%s, model=%s, kind=%s, pages=%s, prompt_hash=%s, prompt_chars=%d, chat_history=%d): %s",
            mode,
            model,
            kind,
            pages,
            prompt_hash,
            prompt_chars,
            chat_history_len,
            prompt_preview,
        )

    def log_retry_exception(
        *,
        mode: str,
        attempt: int,
        max_retries: int,
        model: str | None,
        prompt: str,
        exc: Exception,
        chat_history_len: int = 0,
    ) -> None:
        prompt_hash, prompt_chars, prompt_preview = _prompt_diagnostics(prompt)
        kind = _prompt_kind(prompt)
        pages = _page_refs(prompt)
        logger.warning(
            "Retrying PageIndex DeepSeek JSON completion "
            "(%d/%d, kind=%s, pages=%s, exc=%s, prompt_hash=%s): %s",
            attempt,
            max_retries,
            kind,
            pages,
            type(exc).__name__,
            prompt_hash,
            exc,
        )
        logger.debug(
            "PageIndex DeepSeek JSON exception details "
            "(mode=%s, model=%s, kind=%s, pages=%s, prompt_hash=%s, prompt_chars=%d, chat_history=%d): %s",
            mode,
            model,
            kind,
            pages,
            prompt_hash,
            prompt_chars,
            chat_history_len,
            prompt_preview,
        )

    def llm_completion(model, prompt, chat_history=None, return_finish_reason=False):
        if model:
            model = model.removeprefix("litellm/")
        messages = (
            list(chat_history) + [{"role": "user", "content": prompt}]
            if chat_history
            else [{"role": "user", "content": prompt}]
        )
        max_retries = 10
        last_response = None
        for i in range(max_retries):
            try:
                litellm.drop_params = True
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                last_response = response
                content = _response_content(response)
                if content.strip():
                    if return_finish_reason:
                        finish_reason = (
                            "max_output_reached" if _response_finish_reason(response) == "length" else "finished"
                        )
                        return content, finish_reason
                    return content
                log_empty_response(
                    mode="sync",
                    attempt=i + 1,
                    max_retries=max_retries,
                    model=model,
                    prompt=prompt,
                    response=response,
                    chat_history_len=len(chat_history or []),
                )
            except Exception as exc:
                log_retry_exception(
                    mode="sync",
                    attempt=i + 1,
                    max_retries=max_retries,
                    model=model,
                    prompt=prompt,
                    exc=exc,
                    chat_history_len=len(chat_history or []),
                )
            if i < max_retries - 1:
                time.sleep(1)

        if return_finish_reason:
            finish_reason = (
                "max_output_reached"
                if last_response and _response_finish_reason(last_response) == "length"
                else "finished"
            )
            return "", finish_reason
        return ""

    async def llm_acompletion(model, prompt):
        if model:
            model = model.removeprefix("litellm/")
        messages = [{"role": "user", "content": prompt}]
        max_retries = 10
        for i in range(max_retries):
            try:
                litellm.drop_params = True
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                content = _response_content(response)
                if content.strip():
                    return content
                log_empty_response(
                    mode="async",
                    attempt=i + 1,
                    max_retries=max_retries,
                    model=model,
                    prompt=prompt,
                    response=response,
                )
            except Exception as exc:
                log_retry_exception(
                    mode="async",
                    attempt=i + 1,
                    max_retries=max_retries,
                    model=model,
                    prompt=prompt,
                    exc=exc,
                )
            if i < max_retries - 1:
                import asyncio

                await asyncio.sleep(1)
        return ""

    def toc_detector_single_page(content, model=None):
        try:
            return original_toc_detector(content, model=model)
        except KeyError as exc:
            if exc.args == ("toc_detected",):
                logger.warning(
                    "PageIndex TOC detector returned no toc_detected field; treating page as non-TOC."
                )
                return "no"
            raise

    page_index.llm_completion = llm_completion
    page_index.llm_acompletion = llm_acompletion
    page_index.toc_detector_single_page = toc_detector_single_page

    def restore() -> None:
        page_index.llm_completion = original_llm_completion
        page_index.llm_acompletion = original_llm_acompletion
        page_index.toc_detector_single_page = original_toc_detector

    return restore


def _get_pdf_page_count(pdf_path: Path) -> int:
    from openkb.pdf_parser import get_pdf_page_count

    return get_pdf_page_count(pdf_path)


def _convert_pdf_to_pages(
    pdf_path: Path,
    doc_name: str,
    images_dir: Path,
    mineru_output_dir: Path,
    mineru_backend: str,
) -> list[dict[str, Any]]:
    from openkb.pdf_parser import convert_pdf_to_pages

    return convert_pdf_to_pages(pdf_path, doc_name, images_dir, mineru_output_dir, mineru_backend)


def index_long_document(pdf_path: Path, kb_dir: Path) -> IndexResult:
    """Index a long PDF document using PageIndex and write wiki pages."""
    openkb_dir = kb_dir / ".openkb"
    config = load_config(openkb_dir / "config.yaml")

    model: str = config.get("model", "gpt-5.4")
    mineru_backend: str = config.get("mineru_backend", "hybrid-auto-engine")
    mineru_output_dir = kb_dir / config.get("mineru_output_dir", ".openkb/mineru")
    pageindex_api_key = os.environ.get("PAGEINDEX_API_KEY", "")

    index_config = IndexConfig(
        if_add_node_text=True,
        if_add_node_summary=True,
        if_add_doc_description=True,
    )

    client = PageIndexClient(
        api_key=pageindex_api_key or None,
        model=model,
        storage_path=str(openkb_dir),
        index_config=index_config,
    )
    col = client.collection()

    # Add PDF (retry up to 3 times; PageIndex TOC accuracy is stochastic).
    max_retries = 3
    doc_id = None
    restore_pageindex = _patch_pageindex_deepseek_json_mode(model)
    for attempt in range(1, max_retries + 1):
        try:
            try:
                doc_id = col.add(str(pdf_path))
            finally:
                restore_pageindex()
            logger.info("PageIndex added %s -> doc_id=%s (attempt %d)", pdf_path.name, doc_id, attempt)
            break
        except Exception as exc:
            restore_pageindex()
            logger.warning("PageIndex attempt %d/%d failed for %s: %s", attempt, max_retries, pdf_path.name, exc)
            if attempt == max_retries:
                raise RuntimeError(f"Failed to index {pdf_path.name} after {max_retries} attempts: {exc}") from exc
            restore_pageindex = _patch_pageindex_deepseek_json_mode(model)

    # Fetch complete document (metadata + structure + text).
    doc = col.get_document(doc_id, include_text=True)
    doc_name: str = doc.get("doc_name", pdf_path.stem)
    description: str = doc.get("doc_description", "")
    structure: list = doc.get("structure", [])

    logger.info("Doc keys: %s", list(doc.keys()))
    logger.info("page_count from doc: %s", doc.get("page_count", "NOT PRESENT"))

    tree = {
        "doc_name": doc_name,
        "doc_description": description,
        "structure": structure,
    }

    sources_dir = kb_dir / "wiki" / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    images_dir = sources_dir / "images" / pdf_path.stem

    all_pages: list[dict[str, Any]] = []
    if pageindex_api_key:
        page_count = _get_pdf_page_count(pdf_path)
        try:
            all_pages = _normalize_page_content(col.get_page_content(doc_id, f"1-{page_count}"))
        except Exception as exc:
            logger.warning("Cloud get_page_content failed for %s: %s", pdf_path.name, exc)

    if not all_pages:
        if pageindex_api_key:
            logger.warning("Cloud returned no pages for %s; falling back to MinerU", pdf_path.name)
        all_pages = _normalize_page_content(
            _convert_pdf_to_pages(
                pdf_path,
                pdf_path.stem,
                images_dir,
                mineru_output_dir,
                mineru_backend,
            )
        )

    if not all_pages:
        raise RuntimeError(f"No page content extracted for {pdf_path.name}")

    (sources_dir / f"{pdf_path.stem}.json").write_text(
        json_mod.dumps(all_pages, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    summaries_dir = kb_dir / "wiki" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_md = render_summary_md(tree, pdf_path.stem, doc_id)
    (summaries_dir / f"{pdf_path.stem}.md").write_text(summary_md, encoding="utf-8")

    return IndexResult(doc_id=doc_id, description=description, tree=tree)
