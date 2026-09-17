from __future__ import annotations

from pathlib import Path

import pytest

from apostate import quantize_gguf


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
