"""Tests for the `add` CLI command (Task 10)."""
from __future__ import annotations

import json
import threading
import time
from unittest.mock import patch

from click.testing import CliRunner

from openkb.cli import SUPPORTED_EXTENSIONS, _find_kb_dir, cli


class TestSupportedExtensions:
    def test_pdf_supported(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_md_supported(self):
        assert ".md" in SUPPORTED_EXTENSIONS

    def test_docx_supported(self):
        assert ".docx" in SUPPORTED_EXTENSIONS

    def test_txt_supported(self):
        assert ".txt" in SUPPORTED_EXTENSIONS

    def test_unknown_not_supported(self):
        assert ".xyz" not in SUPPORTED_EXTENSIONS


class TestFindKbDir:
    def test_finds_openkb_dir(self, tmp_path, monkeypatch):
        (tmp_path / ".openkb").mkdir()
        monkeypatch.chdir(tmp_path)
        result = _find_kb_dir()
        assert result is not None

    def test_returns_none_if_no_openkb(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("openkb.cli.load_global_config", return_value={}):
            result = _find_kb_dir()
            assert result is None


class TestAddCommand:
    def _setup_kb(self, tmp_path):
        """Create a minimal KB structure."""
        (tmp_path / "raw").mkdir()
        (tmp_path / "wiki" / "sources" / "images").mkdir(parents=True)
        (tmp_path / "wiki" / "summaries").mkdir(parents=True)
        (tmp_path / "wiki" / "concepts").mkdir(parents=True)
        (tmp_path / "wiki" / "reports").mkdir(parents=True)
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        (openkb_dir / "config.yaml").write_text("model: gpt-4o-mini\n")
        (openkb_dir / "hashes.json").write_text(json.dumps({}))
        return tmp_path

    def test_add_missing_init(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path), \
             patch("openkb.cli._find_kb_dir", return_value=None):
            result = runner.invoke(cli, ["add", "somefile.pdf"])
            assert "No knowledge base found" in result.output

    def test_add_single_file_calls_helper(self, tmp_path):
        kb_dir = self._setup_kb(tmp_path)
        doc = tmp_path / "test.md"
        doc.write_text("# Hello")

        runner = CliRunner()
        with patch("openkb.cli.add_single_file") as mock_add, \
             patch("openkb.cli._find_kb_dir", return_value=kb_dir):
            runner.invoke(cli, ["add", str(doc)])
            mock_add.assert_called_once_with(doc, kb_dir)

    def test_add_directory_uses_batch_runner_for_supported_files(self, tmp_path):
        kb_dir = self._setup_kb(tmp_path)
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "a.md").write_text("# A")
        (docs_dir / "b.txt").write_text("B content")
        (docs_dir / "ignore.xyz").write_text("skip me")

        runner = CliRunner()
        with patch("openkb.cli._add_files_batch") as mock_batch, \
             patch("openkb.cli._find_kb_dir", return_value=kb_dir):
            runner.invoke(cli, ["add", str(docs_dir), "--jobs", "3", "--buffer-size", "4"])
            mock_batch.assert_called_once()
            files = mock_batch.call_args.args[0]
            assert [p.name for p in files] == ["a.md", "b.txt"]
            assert mock_batch.call_args.kwargs["jobs"] == 3
            assert mock_batch.call_args.kwargs["buffer_size"] == 4

    def test_add_directory_uses_configured_jobs_when_option_omitted(self, tmp_path):
        kb_dir = self._setup_kb(tmp_path)
        (kb_dir / ".openkb" / "config.yaml").write_text(
            "model: gpt-4o-mini\nfile_processing_jobs: 7\n"
        )
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "a.md").write_text("# A")

        runner = CliRunner()
        with patch("openkb.cli._add_files_batch") as mock_batch, \
             patch("openkb.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["add", str(docs_dir)])

        assert "Processing with 7 worker(s)." in result.output
        assert mock_batch.call_args.kwargs["jobs"] is None

    def test_add_unsupported_extension(self, tmp_path):
        kb_dir = self._setup_kb(tmp_path)
        doc = tmp_path / "file.xyz"
        doc.write_text("content")

        runner = CliRunner()
        with patch("openkb.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["add", str(doc)])
            assert "Unsupported file type" in result.output

    def test_add_nonexistent_path(self, tmp_path):
        kb_dir = self._setup_kb(tmp_path)

        runner = CliRunner()
        with patch("openkb.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["add", str(tmp_path / "nonexistent.pdf")])
            assert "does not exist" in result.output

    def test_add_skipped_file(self, tmp_path):
        kb_dir = self._setup_kb(tmp_path)
        doc = tmp_path / "test.md"
        doc.write_text("# Hello")

        from openkb.converter import ConvertResult
        mock_result = ConvertResult(skipped=True)

        runner = CliRunner()
        with patch("openkb.cli._find_kb_dir", return_value=kb_dir), \
             patch("openkb.cli.convert_document", return_value=mock_result), \
             patch("openkb.cli.asyncio.run") as mock_arun:
            result = runner.invoke(cli, ["add", str(doc)])
            assert "SKIP" in result.output
            mock_arun.assert_not_called()

    def test_add_short_doc_runs_compiler(self, tmp_path):
        kb_dir = self._setup_kb(tmp_path)
        doc = tmp_path / "test.md"
        doc.write_text("# Hello")

        source_path = kb_dir / "wiki" / "sources" / "test.md"
        source_path.write_text("# Hello converted")

        from openkb.converter import ConvertResult
        mock_result = ConvertResult(
            raw_path=kb_dir / "raw" / "test.md",
            source_path=source_path,
            is_long_doc=False,
        )

        runner = CliRunner()
        with patch("openkb.cli._find_kb_dir", return_value=kb_dir), \
             patch("openkb.cli.convert_document", return_value=mock_result), \
             patch("openkb.cli.asyncio.run") as mock_arun:
            result = runner.invoke(cli, ["add", str(doc)])
            mock_arun.assert_called_once()
            assert "OK" in result.output


class TestAddBatchRunner:
    def _setup_kb(self, tmp_path):
        (tmp_path / "raw").mkdir()
        (tmp_path / "wiki" / "sources").mkdir(parents=True)
        (tmp_path / "wiki" / "summaries").mkdir(parents=True)
        (tmp_path / "wiki" / "concepts").mkdir(parents=True)
        (tmp_path / "wiki" / "log.md").write_text("# Operations Log\n\n")
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        (openkb_dir / "config.yaml").write_text(
            "model: gpt-4o-mini\nfile_processing_jobs: 2\npipeline_buffer_size: 1\n"
        )
        (openkb_dir / "hashes.json").write_text(json.dumps({}))
        return tmp_path

    def test_batch_runner_compiles_and_registers_each_file(self, tmp_path):
        from openkb.cli import _add_files_batch
        from openkb.converter import ConvertResult

        kb_dir = self._setup_kb(tmp_path)
        doc_a = tmp_path / "a.md"
        doc_b = tmp_path / "b.md"
        doc_a.write_text("# A")
        doc_b.write_text("# B")
        source_a = kb_dir / "wiki" / "sources" / "a.md"
        source_b = kb_dir / "wiki" / "sources" / "b.md"
        source_a.write_text("# A converted")
        source_b.write_text("# B converted")

        def convert_side_effect(path, _kb_dir):
            return ConvertResult(
                raw_path=kb_dir / "raw" / path.name,
                source_path=source_a if path.name == "a.md" else source_b,
                is_long_doc=False,
                file_hash=("a" if path.name == "a.md" else "b") * 64,
            )

        async def fake_compile(*_args, **_kwargs):
            return None

        with patch("openkb.cli.convert_document", side_effect=convert_side_effect), \
             patch("openkb.agent.compiler.compile_short_doc", side_effect=fake_compile) as mock_compile:
            counts = _add_files_batch([doc_a, doc_b], kb_dir, jobs=2, buffer_size=1)

        assert counts == {"added": 2, "skipped": 0, "failed": 0}
        assert mock_compile.call_count == 2
        hashes = json.loads((kb_dir / ".openkb" / "hashes.json").read_text())
        assert {meta["name"] for meta in hashes.values()} == {"a.md", "b.md"}

    def test_batch_runner_skips_duplicate_hash_within_batch(self, tmp_path):
        from openkb.cli import _add_files_batch
        from openkb.converter import ConvertResult
        from openkb.state import HashRegistry

        kb_dir = self._setup_kb(tmp_path)
        doc_a = tmp_path / "a.md"
        doc_b = tmp_path / "b.md"
        doc_a.write_text("same")
        doc_b.write_text("same")
        source_a = kb_dir / "wiki" / "sources" / "a.md"
        source_a.write_text("# A converted")
        digest = HashRegistry.hash_file(doc_a)
        converted = ConvertResult(
            raw_path=kb_dir / "raw" / "a.md",
            source_path=source_a,
            is_long_doc=False,
            file_hash=digest,
        )

        async def fake_compile(*_args, **_kwargs):
            return None

        with patch("openkb.cli.convert_document", return_value=converted) as mock_convert, \
             patch("openkb.agent.compiler.compile_short_doc", side_effect=fake_compile):
            counts = _add_files_batch([doc_a, doc_b], kb_dir, jobs=2, buffer_size=1)

        assert counts == {"added": 1, "skipped": 1, "failed": 0}
        mock_convert.assert_called_once()

    def test_batch_runner_rejects_doc_name_conflict_before_conversion(self, tmp_path):
        from openkb.cli import _add_files_batch
        from openkb.converter import ConvertResult

        kb_dir = self._setup_kb(tmp_path)
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        doc_a = dir_a / "paper.md"
        doc_b = dir_b / "paper.md"
        doc_a.write_text("# A")
        doc_b.write_text("# B")
        source_a = kb_dir / "wiki" / "sources" / "paper.md"
        source_a.write_text("# A converted")
        converted = ConvertResult(
            raw_path=kb_dir / "raw" / "paper.md",
            source_path=source_a,
            is_long_doc=False,
            file_hash="a" * 64,
        )

        async def fake_compile(*_args, **_kwargs):
            return None

        with patch("openkb.cli.convert_document", return_value=converted) as mock_convert, \
             patch("openkb.agent.compiler.compile_short_doc", side_effect=fake_compile):
            counts = _add_files_batch([doc_a, doc_b], kb_dir, jobs=2, buffer_size=1)

        assert counts == {"added": 1, "skipped": 0, "failed": 1}
        mock_convert.assert_called_once_with(doc_a, kb_dir)

    def test_batch_runner_rejects_existing_doc_name_conflict(self, tmp_path):
        from openkb.cli import _add_files_batch

        kb_dir = self._setup_kb(tmp_path)
        (kb_dir / ".openkb" / "hashes.json").write_text(json.dumps({
            "b" * 64: {"name": "paper.pdf", "doc_name": "paper", "type": "pdf"}
        }))
        doc = tmp_path / "paper.md"
        doc.write_text("# Different paper")

        with patch("openkb.cli.convert_document") as mock_convert:
            counts = _add_files_batch([doc], kb_dir, jobs=2, buffer_size=1)

        assert counts == {"added": 0, "skipped": 0, "failed": 1}
        mock_convert.assert_not_called()

    def test_add_single_file_rejects_existing_doc_name_conflict(self, tmp_path):
        from openkb.cli import add_single_file
        from openkb.converter import ConvertResult

        kb_dir = self._setup_kb(tmp_path)
        (kb_dir / ".openkb" / "hashes.json").write_text(json.dumps({
            "b" * 64: {"name": "paper.pdf", "doc_name": "paper", "type": "pdf"}
        }))
        doc = tmp_path / "paper.md"
        doc.write_text("# Different paper")
        source_path = kb_dir / "wiki" / "sources" / "paper.md"
        source_path.write_text("# converted")
        converted = ConvertResult(
            raw_path=kb_dir / "raw" / "paper.md",
            source_path=source_path,
            is_long_doc=False,
            file_hash="a" * 64,
        )

        with patch("openkb.cli.convert_document", return_value=converted) as mock_convert, \
             patch("openkb.agent.compiler.compile_short_doc") as mock_compile:
            outcome = add_single_file(doc, kb_dir)

        assert outcome == "failed"
        mock_convert.assert_not_called()
        mock_compile.assert_not_called()


class TestWatchCommand:
    def test_watch_uses_batch_runner_for_debounced_supported_files(self, tmp_path):
        kb_dir = TestAddCommand()._setup_kb(tmp_path)
        raw_dir = kb_dir / "raw"
        doc = raw_dir / "a.md"
        skip = raw_dir / "a.xyz"
        doc.write_text("# A")
        skip.write_text("skip")

        def fake_watch(_raw_dir, callback):
            callback([str(doc), str(skip)])

        runner = CliRunner()
        with patch("openkb.cli._find_kb_dir", return_value=kb_dir), \
             patch("openkb.watcher.watch_directory", side_effect=fake_watch), \
             patch("openkb.cli._add_files_batch") as mock_batch:
            result = runner.invoke(cli, ["watch", "--jobs", "5", "--buffer-size", "6"])

        assert result.exit_code == 0, result.output
        assert "Skipping unsupported file type" in result.output
        mock_batch.assert_called_once()
        assert [p.name for p in mock_batch.call_args.args[0]] == ["a.md"]
        assert mock_batch.call_args.kwargs["jobs"] == 5
        assert mock_batch.call_args.kwargs["buffer_size"] == 6

    def test_watch_serializes_overlapping_debounced_batches(self, tmp_path):
        kb_dir = TestAddCommand()._setup_kb(tmp_path)
        raw_dir = kb_dir / "raw"
        doc_a = raw_dir / "a.md"
        doc_b = raw_dir / "b.md"
        doc_a.write_text("# A")
        doc_b.write_text("# B")

        callbacks = []

        def fake_watch(_raw_dir, callback):
            callbacks.append(callback)

        active = 0
        max_active = 0
        active_lock = threading.Lock()

        def fake_batch(*_args, **_kwargs):
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with active_lock:
                active -= 1

        runner = CliRunner()
        with patch("openkb.cli._find_kb_dir", return_value=kb_dir), \
             patch("openkb.watcher.watch_directory", side_effect=fake_watch), \
             patch("openkb.cli._add_files_batch", side_effect=fake_batch):
            result = runner.invoke(cli, ["watch"])

            assert result.exit_code == 0, result.output
            assert callbacks

            t1 = threading.Thread(target=callbacks[0], args=([str(doc_a)],))
            t2 = threading.Thread(target=callbacks[0], args=([str(doc_b)],))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        assert max_active == 1
