from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")
from gguf import GGUFReader, GGUFValueType, GGUFWriter

from apostate import prepare_quant


HIDDEN = 3
BASE = 4
ALIGNED = 256
LAYERS = 2


def _model_tensors(width: int, *, zero_tail: bool = False) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    cursor = 1.0
    for layer in range(LAYERS + 1):
        current = BASE if layer == LAYERS else width
        for suffix, shape in (
            ("ffn_down", (HIDDEN, current)),
            ("ffn_gate", (current, HIDDEN)),
            ("ffn_up", (current, HIDDEN)),
        ):
            values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + cursor
            cursor += values.size
            if zero_tail and layer < LAYERS and current > BASE:
                values = values.copy()
                if suffix == "ffn_down":
                    values[:, BASE:] = 0
                else:
                    values[BASE:, :] = 0
            tensors[f"blk.{layer}.{suffix}.weight"] = values
        tensors[f"blk.{layer}.attn_q.weight"] = np.arange(9, dtype=np.float32).reshape(3, 3) + cursor
        cursor += 9
    return tensors


def _write_model(path: Path, width: int, *, zero_tail: bool = False) -> Path:
    writer = GGUFWriter(str(path), "test")
    writer.add_uint32("test.feed_forward_length", width)
    writer.add_uint32("test.block_count", LAYERS)
    writer.add_string("general.name", "fixture")
    for name, values in _model_tensors(width, zero_tail=zero_tail).items():
        writer.add_tensor(name, values)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _write_imatrix(path: Path, width: int) -> Path:
    writer = GGUFWriter(str(path), "")
    writer.add_key_value("general.type", "imatrix", GGUFValueType.STRING)
    writer.add_key_value("imatrix.chunk_count", 16, GGUFValueType.UINT32)
    for layer in range(LAYERS):
        name = f"blk.{layer}.ffn_down.weight"
        writer.add_tensor(
            f"{name}.in_sum2", np.arange(1, width + 1, dtype=np.float32).reshape(1, width)
        )
        writer.add_tensor(f"{name}.counts", np.array([32.0], dtype=np.float32))
    writer.add_tensor(
        "blk.0.attn_q.weight.in_sum2",
        np.arange(1, HIDDEN + 1, dtype=np.float32).reshape(1, HIDDEN),
    )
    writer.add_tensor("blk.0.attn_q.weight.counts", np.array([32.0], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _payloads(path: Path) -> dict[str, np.ndarray]:
    return {tensor.name: np.array(tensor.data) for tensor in GGUFReader(str(path)).tensors}


def _shapes(path: Path) -> dict[str, tuple[int, ...]]:
    return {tensor.name: tuple(int(x) for x in tensor.shape) for tensor in GGUFReader(str(path)).tensors}


def test_model_padding_streams_directly_and_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = _write_model(tmp_path / "source.gguf", BASE)
    padded = tmp_path / "padded.gguf"
    stripped = tmp_path / "stripped.gguf"

    def refuse_spool(*_args, **_kwargs):
        raise AssertionError("model-sized temporary spool requested")

    monkeypatch.setattr(tempfile, "SpooledTemporaryFile", refuse_spool)
    receipt = prepare_quant.prepare_model(source, padded, ALIGNED)
    prepare_quant.prepare_model(padded, stripped, BASE)

    assert receipt["direction"] == "pad"
    assert _shapes(padded)["blk.0.ffn_down.weight"] == (ALIGNED, HIDDEN)
    # The dummy (MTP) block moves with the decoder: llama.cpp sizes every block, `nextn` included, from
    # `feed_forward_length`, so a draft head left at the source width makes the file unloadable. The
    # round trip below still returns the source's bytes, because what the draft gains here is zeros.
    assert _shapes(padded)[f"blk.{LAYERS}.ffn_down.weight"] == (ALIGNED, HIDDEN)
    assert receipt["draft_tensors_resized"], "the draft block is reported as resized, not exempt"
    before, after = _payloads(source), _payloads(stripped)
    assert before.keys() == after.keys()
    assert all(np.array_equal(before[name], after[name]) for name in before)


def test_strip_refuses_to_remove_nonzero_weights(tmp_path: Path):
    source = _write_model(tmp_path / "wide.gguf", ALIGNED, zero_tail=False)
    out = tmp_path / "refused.gguf"

    with pytest.raises(prepare_quant.PreparationRefused, match="non-zero"):
        prepare_quant.prepare_model(source, out, BASE)

    assert not out.exists()


def test_imatrix_adaptation_grows_only_matching_statistics(tmp_path: Path):
    source = _write_model(tmp_path / "source.gguf", BASE)
    target = tmp_path / "padded.gguf"
    prepare_quant.prepare_model(source, target, ALIGNED)
    matrix = _write_imatrix(tmp_path / "source.imatrix.gguf", BASE)
    out = tmp_path / "adapted.imatrix.gguf"

    receipt = prepare_quant.adapt_matrix(
        matrix, target, out, expect_append=ALIGNED - BASE
    )

    tensors = _payloads(out)
    assert receipt["entries_grown"] == LAYERS
    assert tensors["blk.0.ffn_down.weight.in_sum2"].shape == (1, ALIGNED)
    assert np.count_nonzero(tensors["blk.0.ffn_down.weight.in_sum2"][:, BASE:]) == 0
    assert tensors["blk.0.attn_q.weight.in_sum2"].shape == (1, HIDDEN)


def test_one_command_prepares_model_and_matrix_with_receipt(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    source = _write_model(tmp_path / "source.gguf", BASE)
    matrix = _write_imatrix(tmp_path / "source.imatrix.gguf", BASE)
    padded = tmp_path / "padded.gguf"
    adapted = tmp_path / "adapted.imatrix.gguf"
    receipt = tmp_path / "receipt.json"

    code = prepare_quant.main([
        "--model", str(source),
        "--out-model", str(padded),
        "--to", str(ALIGNED),
        "--imatrix", str(matrix),
        "--out-imatrix", str(adapted),
        "--expect-append", str(ALIGNED - BASE),
        "--receipt", str(receipt),
    ])

    assert code == 0, capsys.readouterr().err
    document = json.loads(receipt.read_text())
    assert document["model"]["target_width"] == ALIGNED
    assert document["imatrix"]["entries_grown"] == LAYERS
    assert padded.exists() and adapted.exists()


def test_one_command_can_resolve_a_published_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    source = _write_model(tmp_path / "source.gguf", BASE)
    matrix = _write_imatrix(tmp_path / "published.imatrix.gguf", BASE)
    provenance = {
        "publisher": "mradermacher",
        "repository": "mradermacher/Qwen3.8-27B-i1-GGUF",
        "filename": "imatrix.gguf",
    }
    monkeypatch.setattr(
        prepare_quant,
        "resolve_published_imatrix",
        lambda *_args, **_kwargs: (matrix, provenance),
    )

    code = prepare_quant.main([
        "--model", str(source),
        "--out-model", str(tmp_path / "padded.gguf"),
        "--to", str(ALIGNED),
        "--imatrix-source", "mradermacher",
        "--base-model", "Qwen/Qwen3.8-27B",
        "--out-imatrix", str(tmp_path / "adapted.imatrix.gguf"),
        "--expect-append", str(ALIGNED - BASE),
    ])

    assert code == 0, capsys.readouterr().err
    document = json.loads(capsys.readouterr().out)
    assert document["imatrix_source"] == provenance


def test_combined_command_preflights_matrix_before_writing_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    source = _write_model(tmp_path / "source.gguf", BASE)
    matrix = tmp_path / "wrong.imatrix.gguf"
    writer = GGUFWriter(str(matrix), "")
    writer.add_key_value("general.type", "imatrix", GGUFValueType.STRING)
    writer.add_tensor("missing.weight.in_sum2", np.ones(BASE, dtype=np.float32))
    writer.add_tensor("missing.weight.counts", np.ones(1, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    padded = tmp_path / "must-not-exist.gguf"

    code = prepare_quant.main([
        "--model", str(source),
        "--out-model", str(padded),
        "--to", str(ALIGNED),
        "--imatrix", str(matrix),
        "--out-imatrix", str(tmp_path / "adapted.gguf"),
    ])

    assert code == 1
    assert "target model has no tensor" in capsys.readouterr().err
    assert not padded.exists()


def test_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    source = _write_model(tmp_path / "source.gguf", BASE)
    out = tmp_path / "padded.gguf"

    code = prepare_quant.main([
        "--model", str(source), "--out-model", str(out), "--to", str(ALIGNED),
        "--dry-run",
    ])
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["dry_run"] is True
    assert document["model"]["direction"] == "pad"
    assert not out.exists()


def _repo_info(files: dict[str, int], sha: str = "c0ffee"):
    class Sibling:
        def __init__(self, name: str, size: int):
            self.rfilename, self.size = name, size

    class Info:
        def __init__(self):
            self.sha = sha
            self.siblings = [Sibling(name, size) for name, size in files.items()]

    return Info()


def test_bartowski_matrix_is_discovered_instead_of_assuming_one_filename(tmp_path: Path):
    attempts: list[str] = []

    class Api:
        def repo_info(self, repo_id: str, **_kwargs):
            attempts.append(repo_id)
            if repo_id == "bartowski/Qwen3.8-27B-GGUF":
                return _repo_info({"README.md": 10, "Qwen3.8-27B-imatrix.gguf": 4096})
            raise AssertionError(f"unexpected repository {repo_id}")

    def download(*, repo_id: str, filename: str, revision: str, local_dir: str):
        assert revision == "c0ffee", "the download must be pinned to a resolved commit"
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"matrix")
        return str(destination)

    path, provenance = prepare_quant.resolve_published_imatrix(
        "bartowski", "Qwen/Qwen3.8-27B", tmp_path, api=Api(), downloader=download
    )

    assert path.read_bytes() == b"matrix"
    assert attempts == ["bartowski/Qwen3.8-27B-GGUF"]
    assert provenance["repository"] == "bartowski/Qwen3.8-27B-GGUF"
    assert provenance["filename"] == "Qwen3.8-27B-imatrix.gguf"
    assert provenance["revision"] == "c0ffee"
    assert provenance["sha256"] == hashlib.sha256(b"matrix").hexdigest()
    assert provenance["origin"] == "hub"


def test_published_matrix_rejects_cache_path_traversal(tmp_path: Path):
    with pytest.raises(prepare_quant.PreparationRefused, match="valid owner/model"):
        prepare_quant.resolve_published_imatrix(
            "mradermacher", "../../outside", tmp_path, api=object()
        )


def test_published_matrix_cache_avoids_a_hub_call(tmp_path: Path):
    cached = tmp_path / "mradermacher" / "Qwen3.8-27B-i1-GGUF" / "Qwen3.8-27B.imatrix.gguf"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached")

    class OfflineApi:
        def repo_info(self, *_args, **_kwargs):
            raise AssertionError("cache hit attempted a Hub request")

    path, provenance = prepare_quant.resolve_published_imatrix(
        "mradermacher", "Qwen/Qwen3.8-27B", tmp_path, api=OfflineApi()
    )

    assert path == cached.resolve()
    assert provenance["repository"] == "mradermacher/Qwen3.8-27B-i1-GGUF"
    assert provenance["origin"] == "cache"
    assert provenance["revision"] is None, "a cached file cannot claim a verified commit"
    assert provenance["sha256"] == hashlib.sha256(b"cached").hexdigest()


def test_a_symlinked_cache_entry_is_ignored(tmp_path: Path):
    cache = tmp_path / "mradermacher" / "Qwen3.8-27B-i1-GGUF"
    cache.mkdir(parents=True)
    outside = tmp_path / "elsewhere.imatrix.gguf"
    outside.write_bytes(b"planted")
    (cache / "imatrix.gguf").symlink_to(outside)

    class Api:
        def repo_info(self, *_args, **_kwargs):
            raise LookupError("no such repository")

    with pytest.raises(prepare_quant.PreparationRefused, match="no mradermacher imatrix"):
        prepare_quant.resolve_published_imatrix(
            "mradermacher", "Qwen/Qwen3.8-27B", tmp_path, api=Api(), downloader=lambda **_k: None
        )


def test_a_hostile_remote_filename_is_refused(tmp_path: Path):
    class Api:
        def repo_info(self, *_args, **_kwargs):
            return _repo_info({"../../escaped-imatrix.gguf": 4096})

    def download(**_kwargs):
        raise AssertionError("download attempted for an unsafe remote filename")

    with pytest.raises(prepare_quant.PreparationRefused, match="unsafe remote filename"):
        prepare_quant.resolve_published_imatrix(
            "mradermacher", "Qwen/Qwen3.8-27B", tmp_path, api=Api(), downloader=download
        )


def test_an_implausibly_large_remote_imatrix_is_refused(tmp_path: Path):
    class Api:
        def repo_info(self, *_args, **_kwargs):
            return _repo_info({"imatrix.gguf": prepare_quant.IMATRIX_MAX_BYTES + 1})

    def download(**_kwargs):
        raise AssertionError("download attempted for an oversized file")

    with pytest.raises(prepare_quant.PreparationRefused, match="too large"):
        prepare_quant.resolve_published_imatrix(
            "mradermacher", "Qwen/Qwen3.8-27B", tmp_path, api=Api(), downloader=download
        )


def test_a_download_outside_the_cache_is_refused(tmp_path: Path):
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"matrix")

    class Api:
        def repo_info(self, *_args, **_kwargs):
            return _repo_info({"imatrix.gguf": 4096})

    def download(**_kwargs):
        return str(outside)

    with pytest.raises(prepare_quant.PreparationRefused, match="outside the cache"):
        prepare_quant.resolve_published_imatrix(
            "mradermacher", "Qwen/Qwen3.8-27B", tmp_path, api=Api(), downloader=download
        )


def test_artifacts_record_only_basenames_not_local_paths(tmp_path: Path):
    source = _write_model(tmp_path / "source.gguf", BASE)
    out = tmp_path / "padded.gguf"
    prepare_quant.prepare_model(source, out, ALIGNED)

    fields = {k: v.contents() for k, v in GGUFReader(str(out)).fields.items()}
    recorded = fields["apostate.quant_preparation.source"]
    assert recorded == "source.gguf"
    assert str(tmp_path) not in str(recorded), "a published artifact must not carry local paths"


def test_an_interrupted_run_leaves_an_explained_stub(tmp_path: Path):
    source = _write_model(tmp_path / "source.gguf", BASE)
    out = tmp_path / "padded.gguf"
    out.touch()

    with pytest.raises(prepare_quant.PreparationRefused, match="interrupted"):
        prepare_quant.prepare_model(source, out, ALIGNED)


def test_an_unwritable_output_directory_is_a_refusal_not_a_traceback(tmp_path: Path):
    source = _write_model(tmp_path / "source.gguf", BASE)

    with pytest.raises(prepare_quant.PreparationRefused, match="output directory"):
        prepare_quant.prepare_model(source, tmp_path / "absent" / "padded.gguf", ALIGNED)


def test_expect_append_must_match_exactly(tmp_path: Path):
    source = _write_model(tmp_path / "source.gguf", BASE)
    target = tmp_path / "padded.gguf"
    prepare_quant.prepare_model(source, target, ALIGNED)
    matrix = _write_imatrix(tmp_path / "wrong.imatrix.gguf", BASE - 1)

    with pytest.raises(prepare_quant.PreparationRefused, match="does not equal"):
        prepare_quant.adapt_matrix(
            matrix, target, tmp_path / "adapted.gguf", expect_append=ALIGNED - BASE
        )


def test_matching_imatrix_still_creates_the_requested_output(tmp_path: Path):
    target = _write_model(tmp_path / "target.gguf", ALIGNED, zero_tail=True)
    matrix = _write_imatrix(tmp_path / "matching.imatrix.gguf", ALIGNED)
    out = tmp_path / "copied.imatrix.gguf"

    receipt = prepare_quant.adapt_matrix(matrix, target, out)

    assert receipt["entries_grown"] == 0
    assert out.exists()


def test_receipt_cannot_overwrite_an_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    source = _write_model(tmp_path / "source.gguf", BASE)

    code = prepare_quant.main([
        "--model", str(source),
        "--out-model", str(tmp_path / "padded.gguf"),
        "--to", str(ALIGNED),
        "--receipt", str(source),
    ])

    assert code == 1
    assert "receipt" in capsys.readouterr().err
    assert GGUFReader(str(source)).tensors


def test_combined_failure_removes_outputs_created_by_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = _write_model(tmp_path / "source.gguf", BASE)
    matrix = _write_imatrix(tmp_path / "source.imatrix.gguf", BASE)
    padded = tmp_path / "padded.gguf"
    adapted = tmp_path / "adapted.gguf"

    def fail_adaptation(*_args, **_kwargs):
        raise prepare_quant.PreparationRefused("injected matrix failure")

    monkeypatch.setattr(prepare_quant, "adapt_matrix", fail_adaptation)
    code = prepare_quant.main([
        "--model", str(source),
        "--out-model", str(padded),
        "--to", str(ALIGNED),
        "--imatrix", str(matrix),
        "--out-imatrix", str(adapted),
    ])

    assert code == 1
    assert not padded.exists()
    assert not adapted.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="creating a symlink needs privilege on Windows")
def test_dangling_output_symlink_is_not_followed(tmp_path: Path):
    source = _write_model(tmp_path / "source.gguf", BASE)
    out = tmp_path / "output.gguf"
    victim = tmp_path / "victim.gguf"
    out.symlink_to(victim)

    with pytest.raises(prepare_quant.PreparationRefused, match="already exists"):
        prepare_quant.prepare_model(source, out, ALIGNED)

    assert not victim.exists(), "the symlink target must never be created"


def test_published_output_is_a_plain_file_on_any_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Publishing must not depend on hard links: exFAT has none and Windows os.link rejects them."""
    source = _write_model(tmp_path / "source.gguf", BASE)
    out = tmp_path / "padded.gguf"

    def refuse_link(*_args, **_kwargs):
        raise AssertionError("publishing attempted a hard link")

    monkeypatch.setattr(os, "link", refuse_link)
    prepare_quant.prepare_model(source, out, ALIGNED)

    assert out.is_file() and not out.is_symlink()
    assert out.stat().st_nlink == 1

