from __future__ import annotations

import json
from pathlib import Path

import pytest

from apostate import quantize_gguf

np = pytest.importorskip("numpy")
gguf = pytest.importorskip("gguf")
from gguf import GGUFWriter  # noqa: E402


def test_explicit_quantizer_wins_over_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    explicit = tmp_path / "llama-quantize"
    explicit.write_text("", encoding="utf-8")
    monkeypatch.setattr(quantize_gguf.shutil, "which", lambda _name: None)

    assert quantize_gguf.resolve_quantizer(str(explicit)) == explicit


def test_missing_quantizer_is_a_clear_refusal(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(quantize_gguf.shutil, "which", lambda _name: None)

    with pytest.raises(quantize_gguf.QuantizationRefused, match="llama-quantize"):
        quantize_gguf.resolve_quantizer("")


def test_command_forwards_imatrix_and_tensor_pins(tmp_path: Path):
    quantizer = tmp_path / "llama-quantize"
    source = tmp_path / "source.gguf"
    output = tmp_path / "output.gguf"
    matrix = tmp_path / "matrix.gguf"

    command = quantize_gguf.command(
        quantizer, source, output, "Q4_K_M", matrix,
        ["blk.64.ffn_down.weight:Q8_0", "token_embd.weight:Q6_K"], 8,
    )

    assert command == [
        str(quantizer), "--imatrix", str(matrix),
        "--tensor-type", "blk.64.ffn_down.weight=Q8_0",
        "--tensor-type", "token_embd.weight=Q6_K",
        str(source), str(output), "q4_k_m", "8",
    ]


def test_command_receipt_is_json_argv_not_shell_text(tmp_path: Path):
    rendered = quantize_gguf.render_command([str(tmp_path / "two words"), "line\nbreak"])

    assert rendered == '["' + str(tmp_path / "two words") + '", "line\\nbreak"]'


# --- the MTP pin ----------------------------------------------------------------------------------------


def _mtp_gguf(path: Path, *, block_count: int, nextn: int | None, blocks: int) -> Path:
    """A converted-GGUF stand-in: what the header declares, and which blocks the file actually holds."""
    writer = GGUFWriter(str(path), "qwen35")
    writer.add_block_count(block_count)
    if nextn is not None:
        writer.add_nextn_predict_layers(nextn)
    for layer in range(blocks):
        writer.add_tensor(f"blk.{layer}.ffn_down.weight", np.ones((8, 8), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _quantizer(tmp_path: Path) -> Path:
    """A stand-in for the external binary: resolution only asks whether the path is a file."""
    path = tmp_path / "llama-quantize"
    path.write_text("", encoding="utf-8")
    return path


def test_mtp_pin_reads_the_draft_index_from_the_files_own_header(tmp_path: Path):
    """`block_count` spells the draft two ways, and the pin must land on the same block either way.

    65 blocks with one MTP layer counts the draft layer as a block, so the draft is 64; 64 blocks with
    the draft layer at 64 is the same block as the count left to the trunk. The index is a property of
    this file, so a hand-written `blk.64.*` is right for one model and silently wrong for the next.
    """
    counted = _mtp_gguf(tmp_path / "counted.gguf", block_count=65, nextn=1, blocks=65)
    uncounted = _mtp_gguf(tmp_path / "uncounted.gguf", block_count=64, nextn=None, blocks=65)

    assert quantize_gguf.mtp_pin(counted, "Q8_0") == "blk.64.*:Q8_0"
    assert quantize_gguf.mtp_pin(uncounted, "Q8_0") == "blk.64.*:Q8_0"


def test_the_flag_composes_the_derived_pin_after_an_explicit_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """The derived pin reaches the argv, and an explicit `--tensor-type` is composed before it.

    llama-quantize applies the *first* `--tensor-type` pattern that matches a tensor name and stops, so
    the narrow, deliberate pin for a draft tensor wins while the derived `blk.<draft>.*` still covers the
    block's other tensors -- a default, not an override.
    """
    source = _mtp_gguf(tmp_path / "source.gguf", block_count=65, nextn=1, blocks=65)
    out = tmp_path / "out.gguf"

    code = quantize_gguf.main([
        "--source", str(source), "--out", str(out), "--quantization", "Q4_K_M",
        "--quantizer", str(_quantizer(tmp_path)),
        "--tensor-type", "blk.64.ffn_down.weight:Q6_K", "--mtp-quantization", "Q8_0", "--dry-run",
    ])
    argv = json.loads(capsys.readouterr().out.removeprefix("+ ").strip())

    assert code == 0
    assert [argv[index + 1] for index, value in enumerate(argv) if value == "--tensor-type"] == [
        "blk.64.ffn_down.weight=Q6_K", "blk.64.*=Q8_0",
    ]
    assert argv[-4:] == [str(source), str(out), "q4_k_m", "0"]
    assert not out.exists(), "--dry-run writes nothing"


def test_a_trunk_only_source_is_refused_by_name(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """The silence this flag exists to prevent: a pin that matches no tensor.

    llama-quantize reports a `--tensor-type` pattern that matched nothing with complete silence -- exit
    0, no warning, quantized as if the flag had not been passed -- so the pin is checked against the
    source's own tensor names before it is composed into an argv.
    """
    trunk = _mtp_gguf(tmp_path / "trunk-only.gguf", block_count=64, nextn=None, blocks=64)
    out = tmp_path / "out.gguf"

    code = quantize_gguf.main([
        "--source", str(trunk), "--out", str(out), "--quantization", "Q4_K_M",
        "--quantizer", str(_quantizer(tmp_path)), "--mtp-quantization", "Q8_0",
    ])
    err = capsys.readouterr().err

    assert code == 1
    assert "quantize-gguf: refused:" in err
    assert "pins blk.64.*" in err, err
    assert "no tensor in block 64" in err, err
    assert not out.exists()


def test_a_source_that_is_not_gguf_is_refused_as_this_commands_own_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Reading the header is this command's first look at the file, so its failure is this command's."""
    source = tmp_path / "not.gguf"
    source.write_bytes(b"not a gguf at all")

    code = quantize_gguf.main([
        "--source", str(source), "--out", str(tmp_path / "out.gguf"), "--quantization", "Q4_K_M",
        "--quantizer", str(_quantizer(tmp_path)), "--mtp-quantization", "Q8_0",
    ])
    err = capsys.readouterr().err

    assert code == 1
    assert "quantize-gguf: refused:" in err, err
    assert "as GGUF" in err, err
