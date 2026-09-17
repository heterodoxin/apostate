"""What `apostate convert-tree` writes from a bake's own tree, and what it refuses.

The bake's output is a sharded safetensors tree whose `intermediate_size` is `base + 1` -- one appended
neuron per band layer -- so nothing downstream can quantize it until that width is repaired. This module
is the step that turns the tree into a GGUF *and* repairs the width in the same pass, which means the
things worth pinning are exactly the ones a reader cannot check by eye:

* the three per-architecture transforms (a norm's `w - 1` storage, `ssm_a`'s `-exp(A_log)`, the
  grouped-to-tiled V-head order), each of which produces a file that loads happily and is wrong;
* the computed pad target, including that the draft (MTP) block moves with the decoder, because
  llama.cpp sizes every block from `feed_forward_length`;
* the draft block's `nextn` names and the block/nextn/recurrent metadata that go with them;
* the vocabulary's source of truth, which is `tokenizer_config.json` before `config.json`.

The fixtures are synthetic and tiny, so the suite needs no model and no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
gguf = pytest.importorskip("gguf")
from gguf import GGUFReader, TokenType  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apostate import convert_tree  # noqa: E402

HIDDEN = 8
WIDTH = 5            # the "17409": one appended neuron, deliberately not a block multiple
DRAFT_WIDTH = 4      # the "17408": the head the additive edit never touches
ALIGNED = 256
LAYERS = 2
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 4
VOCAB = 8
# The SSM geometry: two K heads, four V heads of two, so the V-head order is a real permutation.
KEY_HEADS = 2
VALUE_HEADS = 4
KEY_DIM = 2
VALUE_DIM = 2
QK_ROWS = 2 * KEY_HEADS * KEY_DIM
V_ROWS = VALUE_HEADS * VALUE_DIM
TILED = [0, 1, 4, 5, 2, 3, 6, 7]        # grouped -> tiled, 2 K heads x 2 V heads of 2
TILED_UNIT = [0, 2, 1, 3]               # the same permutation where a V head is one element wide


def _bf16(*shape: int, start: float = 1.0):
    """Distinct non-zero values, so a moved or zeroed byte is visible."""
    count = int(np.prod(shape))
    return (torch.arange(count, dtype=torch.float32).reshape(shape) + start).to(torch.bfloat16)


def _config() -> dict:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": LAYERS,
            "hidden_size": HIDDEN,
            "intermediate_size": WIDTH,
            "max_position_embeddings": 128,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "rms_norm_eps": 1e-06,
            "partial_rotary_factor": 0.5,
            "full_attention_interval": 2,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": KEY_DIM,
            "linear_num_key_heads": KEY_HEADS,
            "linear_num_value_heads": VALUE_HEADS,
            "linear_value_head_dim": VALUE_DIM,
            "vocab_size": VOCAB,
            # Deliberately *not* the id the tokenizer config names: upstream resolves a special token by
            # name from `tokenizer_config.json` first and only then by id from here.
            "bos_token_id": 0,
            "eos_token_id": 1,
            "mtp_num_hidden_layers": 1,
            "rope_parameters": {"rope_theta": 1000000, "mrope_section": [1, 1, 1]},
        },
    }


def _tree(path: Path, *, with_draft: bool = True, width: int = WIDTH) -> Path:
    """A miniature of the real family: two decoder layers, a draft head at a *different* width."""
    path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, object] = {
        "model.language_model.embed_tokens.weight": _bf16(VOCAB, HIDDEN),
        "model.language_model.norm.weight": _bf16(HIDDEN),
        "lm_head.weight": _bf16(VOCAB, HIDDEN, start=50.0),
    }
    for layer in range(LAYERS):
        prefix = f"model.language_model.layers.{layer}"
        tensors[f"{prefix}.input_layernorm.weight"] = _bf16(HIDDEN)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = _bf16(HIDDEN)
        tensors[f"{prefix}.mlp.gate_proj.weight"] = _bf16(width, HIDDEN)
        tensors[f"{prefix}.mlp.up_proj.weight"] = _bf16(width, HIDDEN)
        tensors[f"{prefix}.mlp.down_proj.weight"] = _bf16(HIDDEN, width)
        if (layer + 1) % 2 == 0:      # full attention on every second layer
            tensors[f"{prefix}.self_attn.q_proj.weight"] = _bf16(HEADS * HEAD_DIM, HIDDEN)
            tensors[f"{prefix}.self_attn.k_proj.weight"] = _bf16(KV_HEADS * HEAD_DIM, HIDDEN)
            tensors[f"{prefix}.self_attn.v_proj.weight"] = _bf16(KV_HEADS * HEAD_DIM, HIDDEN)
            tensors[f"{prefix}.self_attn.o_proj.weight"] = _bf16(HIDDEN, HEADS * HEAD_DIM)
            tensors[f"{prefix}.self_attn.q_norm.weight"] = _bf16(HEAD_DIM)
            tensors[f"{prefix}.self_attn.k_norm.weight"] = _bf16(HEAD_DIM)
        else:
            tensors[f"{prefix}.linear_attn.in_proj_qkv.weight"] = _bf16(QK_ROWS + V_ROWS, HIDDEN)
            tensors[f"{prefix}.linear_attn.in_proj_z.weight"] = _bf16(V_ROWS, HIDDEN)
            tensors[f"{prefix}.linear_attn.out_proj.weight"] = _bf16(HIDDEN, V_ROWS)
            tensors[f"{prefix}.linear_attn.in_proj_a.weight"] = _bf16(VALUE_HEADS, HIDDEN)
            tensors[f"{prefix}.linear_attn.in_proj_b.weight"] = _bf16(VALUE_HEADS, HIDDEN)
            tensors[f"{prefix}.linear_attn.conv1d.weight"] = _bf16(QK_ROWS + V_ROWS, 1, 4)
            tensors[f"{prefix}.linear_attn.norm.weight"] = _bf16(VALUE_DIM)
            tensors[f"{prefix}.linear_attn.A_log"] = _bf16(VALUE_HEADS)
            tensors[f"{prefix}.linear_attn.dt_bias"] = _bf16(VALUE_HEADS)
    if with_draft:
        tensors["mtp.fc.weight"] = _bf16(HIDDEN, 2 * HIDDEN)
        tensors["mtp.norm.weight"] = _bf16(HIDDEN)
        tensors["mtp.pre_fc_norm_embedding.weight"] = _bf16(HIDDEN)
        tensors["mtp.pre_fc_norm_hidden.weight"] = _bf16(HIDDEN)
        tensors["mtp.layers.0.input_layernorm.weight"] = _bf16(HIDDEN)
        tensors["mtp.layers.0.post_attention_layernorm.weight"] = _bf16(HIDDEN)
        tensors["mtp.layers.0.mlp.gate_proj.weight"] = _bf16(DRAFT_WIDTH, HIDDEN)
        tensors["mtp.layers.0.mlp.up_proj.weight"] = _bf16(DRAFT_WIDTH, HIDDEN)
        tensors["mtp.layers.0.mlp.down_proj.weight"] = _bf16(HIDDEN, DRAFT_WIDTH)
        tensors["mtp.layers.0.self_attn.q_proj.weight"] = _bf16(HEADS * HEAD_DIM, HIDDEN)
        tensors["mtp.layers.0.self_attn.k_proj.weight"] = _bf16(KV_HEADS * HEAD_DIM, HIDDEN)
        tensors["mtp.layers.0.self_attn.v_proj.weight"] = _bf16(KV_HEADS * HEAD_DIM, HIDDEN)
        tensors["mtp.layers.0.self_attn.o_proj.weight"] = _bf16(HIDDEN, HEADS * HEAD_DIM)
        tensors["mtp.layers.0.self_attn.q_norm.weight"] = _bf16(HEAD_DIM)
        tensors["mtp.layers.0.self_attn.k_norm.weight"] = _bf16(HEAD_DIM)

    save_file(tensors, str(path / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1},
                    "weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors}}),
        encoding="utf-8",
    )
    (path / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (path / "generation_config.json").write_text(
        json.dumps({"top_k": 20, "top_p": 0.95, "temperature": 1.0}), encoding="utf-8"
    )
    (path / "tokenizer.json").write_text(
        json.dumps({
            "model": {"vocab": {"a": 0, "b": 1, "c": 2, "d": 3}, "merges": [["a", "b"], ["c", "d"]]},
            "normalizer": {"type": "NFC"},
            "added_tokens": [
                {"id": 4, "content": "<|endoftext|>", "special": True, "normalized": False},
                {"id": 5, "content": "<|im_end|>", "special": False, "normalized": False},
            ],
        }),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"}), encoding="utf-8"
    )
    return path


def _payloads(path: Path) -> dict[str, np.ndarray]:
    """Payloads keyed by name. The reader hands BF16 back as raw uint8, so re-view it as uint16."""
    out: dict[str, np.ndarray] = {}
    for tensor in GGUFReader(str(path)).tensors:
        array = np.array(tensor.data)
        if tensor.tensor_type.name == "BF16" and array.dtype == np.uint8:
            array = np.ascontiguousarray(array).view(np.uint16)
        out[tensor.name] = array
    return out


def _shapes(path: Path) -> dict[str, tuple[int, ...]]:
    return {tensor.name: tuple(int(n) for n in tensor.shape) for tensor in GGUFReader(str(path)).tensors}


def _fields(path: Path) -> dict:
    return {key: field.contents() for key, field in GGUFReader(str(path)).fields.items()}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    return _tree(tmp_path / "bake-tree")


def test_bf16_payloads_arrive_bit_for_bit(tmp_path: Path, tree: Path):
    """The speed argument rests on this: BF16 out of bf16 is a reinterpretation, not a conversion."""
    out = tmp_path / "model.gguf"
    convert_tree.convert(tree, out, no_pad=True)

    after = _payloads(out)
    assert np.array_equal(after["blk.0.ffn_gate.weight"], _bf16(WIDTH, HIDDEN).view(torch.uint16).numpy())
    assert np.array_equal(after["token_embd.weight"], _bf16(VOCAB, HIDDEN).view(torch.uint16).numpy())


def test_the_pad_target_is_computed_from_the_tree(tmp_path: Path, tree: Path):
    """Nobody types 17664: the bake's 17409 is one past a block multiple, so the tool pads to the next."""
    out = tmp_path / "auto.gguf"
    receipt = convert_tree.convert(tree, out, with_mtp=True)

    assert receipt.source_width == WIDTH
    assert receipt.target_width == ALIGNED
    assert receipt.padded == "auto"
    assert _shapes(out)["blk.0.ffn_gate.weight"] == (HIDDEN, ALIGNED)
    assert any("padded" in warning for warning in receipt.warnings)


def test_the_draft_head_moves_with_the_decoder(tmp_path: Path, tree: Path):
    """llama.cpp sizes every block, `nextn` included, from `feed_forward_length`: a draft head left at
    its own width makes the file unloadable, so the requested width binds there too."""
    out = tmp_path / "draft.gguf"
    receipt = convert_tree.convert(tree, out, with_mtp=True)

    shapes = _shapes(out)
    assert shapes[f"blk.{LAYERS}.ffn_down.weight"] == (ALIGNED, HIDDEN)
    assert shapes[f"blk.{LAYERS}.ffn_gate.weight"] == (HIDDEN, ALIGNED)
    assert receipt.padded_tensors == (LAYERS + 1) * 3
    # the head's own weights survive; only the appended region is zero
    gate = _payloads(out)[f"blk.{LAYERS}.ffn_gate.weight"]
    assert np.array_equal(gate[:DRAFT_WIDTH, :], _bf16(DRAFT_WIDTH, HIDDEN).view(torch.uint16).numpy())
    assert np.count_nonzero(gate[DRAFT_WIDTH:, :]) == 0


def test_an_aligned_tree_is_not_padded(tmp_path: Path):
    """Declaration and tensors agreeing at a block-aligned width is the no-op case."""
    tree = _tree(tmp_path / "aligned", width=ALIGNED)
    config = json.loads((tree / "config.json").read_text())
    config["text_config"]["intermediate_size"] = ALIGNED
    (tree / "config.json").write_text(json.dumps(config), encoding="utf-8")
    receipt = convert_tree.convert(tree, tmp_path / "aligned.gguf")

    assert receipt.padded == "none" and receipt.target_width == ALIGNED
    assert receipt.padded_tensors == 0
    assert not [warning for warning in receipt.warnings if "padded" in warning]


def test_no_pad_records_the_fallback_it_causes(tmp_path: Path, tree: Path):
    receipt = convert_tree.convert(tree, tmp_path / "raw.gguf", no_pad=True)

    assert receipt.padded == "none" and receipt.target_width == WIDTH
    assert any("F16" in warning for warning in receipt.warnings)


def test_the_norms_are_written_as_w_plus_one(tmp_path: Path, tree: Path):
    """This family stores `w - 1`; a GGUF with an unshifted norm loads happily and is silently wrong."""
    out = tmp_path / "norms.gguf"
    convert_tree.convert(tree, out, no_pad=True)

    after = _payloads(out)
    expected = torch.from_numpy(_bf16(HIDDEN).float().numpy() + 1.0).numpy()
    assert np.array_equal(after["blk.0.attn_norm.weight"], expected)
    assert np.array_equal(after["output_norm.weight"], expected)
    # ... and the gated SSM norm is the one norm that already stores its weight
    assert np.array_equal(after["blk.0.ssm_norm.weight"], _bf16(VALUE_DIM).float().numpy())


def test_ssm_a_is_the_negated_exponential_of_a_log(tmp_path: Path, tree: Path):
    out = tmp_path / "ssma.gguf"
    convert_tree.convert(tree, out, no_pad=True)

    a_log = _bf16(VALUE_HEADS).float().numpy()
    tiled = torch.from_numpy(np.ascontiguousarray(a_log[TILED_UNIT].astype(np.float32)))
    after = _payloads(out)
    assert np.array_equal(after["blk.0.ssm_a"], (-torch.exp(tiled)).numpy())
    assert not np.array_equal(after["blk.0.ssm_a"], a_log), "A_log was copied through"


def test_the_v_heads_are_reordered_from_grouped_to_tiled(tmp_path: Path, tree: Path):
    out = tmp_path / "tiled.gguf"
    convert_tree.convert(tree, out, no_pad=True)
    after = _payloads(out)

    qkv = _bf16(QK_ROWS + V_ROWS, HIDDEN).view(torch.uint16).numpy()
    assert np.array_equal(after["blk.0.attn_qkv.weight"],
                          np.concatenate([qkv[:QK_ROWS], qkv[QK_ROWS:][TILED]]))
    gate = _bf16(V_ROWS, HIDDEN).view(torch.uint16).numpy()
    assert np.array_equal(after["blk.0.attn_gate.weight"], gate[TILED])
    alpha = _bf16(VALUE_HEADS, HIDDEN).view(torch.uint16).numpy()
    assert np.array_equal(after["blk.0.ssm_alpha.weight"], alpha[TILED_UNIT])
    assert np.array_equal(after["blk.0.ssm_beta.weight"], alpha[TILED_UNIT])
    out_proj = _bf16(HIDDEN, V_ROWS).view(torch.uint16).numpy()
    assert np.array_equal(after["blk.0.ssm_out.weight"], out_proj[:, TILED])
    conv = _bf16(QK_ROWS + V_ROWS, 1, 4).squeeze(1).float().numpy()
    assert np.array_equal(after["blk.0.ssm_conv1d.weight"],
                          np.concatenate([conv[:QK_ROWS], conv[QK_ROWS:][TILED]]))


def test_the_draft_head_takes_the_nextn_names(tmp_path: Path, tree: Path):
    out = tmp_path / "nextn.gguf"
    convert_tree.convert(tree, out, with_mtp=True)

    shapes = _shapes(out)
    for tail in ("eh_proj", "enorm", "hnorm", "shared_head_norm"):
        assert f"blk.{LAYERS}.nextn.{tail}.weight" in shapes
    assert not [name for name in shapes if name.startswith("mtp.")]
    fields = _fields(out)
    assert fields["qwen35.block_count"] == LAYERS + 1
    assert fields["qwen35.nextn_predict_layers"] == 1
    assert list(fields["qwen35.attention.recurrent_layers"]) == [True, False, False]
    assert fields["qwen35.feed_forward_length"] == ALIGNED


def test_the_draft_block_is_excluded_by_default(tmp_path: Path, tree: Path):
    out = tmp_path / "trunk.gguf"
    receipt = convert_tree.convert(tree, out)

    assert receipt.draft_tensors == 0
    assert not [name for name in _shapes(out) if name.startswith("mtp.")]
    assert f"blk.{LAYERS}.ffn_down.weight" not in _shapes(out)


def test_the_vocabulary_comes_from_the_tokenizer_config_first(tmp_path: Path, tree: Path):
    out = tmp_path / "vocab.gguf"
    convert_tree.convert(tree, out, no_pad=True)

    fields = _fields(out)
    assert len(list(fields["tokenizer.ggml.tokens"])) == VOCAB
    assert list(fields["tokenizer.ggml.tokens"])[:6] == [
        "a", "b", "c", "d", "<|endoftext|>", "<|im_end|>"
    ]
    # ids the tokenizer never defines are placeholders, and look-special added tokens are control tokens
    assert list(fields["tokenizer.ggml.token_type"])[6:] == [TokenType.UNUSED] * 2
    assert fields["tokenizer.ggml.token_type"][5] == TokenType.CONTROL
    # config.json says eos 1; the tokenizer config names `<|im_end|>`, which is id 5
    assert fields["tokenizer.ggml.eos_token_id"] == 5
    assert fields["tokenizer.ggml.padding_token_id"] == 4


def test_an_unsupported_architecture_is_refused(tmp_path: Path):
    tree = _tree(tmp_path / "other")
    config = _config()
    config["model_type"] = "llama"
    config.pop("text_config")
    (tree / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(convert_tree.ConversionRefused, match="not supported"):
        convert_tree.convert(tree, tmp_path / "o.gguf")


def test_an_unmapped_tensor_is_refused_rather_than_dropped(tmp_path: Path):
    tree = _tree(tmp_path / "surprise")
    index = json.loads((tree / "model.safetensors.index.json").read_text())
    index["weight_map"]["model.language_model.layers.0.mlp.mystery.weight"] = "model-00001-of-00001.safetensors"
    (tree / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(convert_tree.ConversionRefused, match="no mapping for tensor"):
        convert_tree.convert(tree, tmp_path / "o.gguf")


def test_a_missing_shard_is_refused(tmp_path: Path):
    tree = _tree(tmp_path / "broken")
    (tree / "model-00001-of-00001.safetensors").unlink()

    with pytest.raises(convert_tree.ConversionRefused, match="names a shard"):
        convert_tree.convert(tree, tmp_path / "o.gguf")


def test_with_mtp_on_a_tree_without_a_draft_head_is_refused(tmp_path: Path):
    tree = _tree(tmp_path / "no-draft", with_draft=False)

    with pytest.raises(convert_tree.ConversionRefused, match="holds no mtp."):
        convert_tree.convert(tree, tmp_path / "o.gguf", with_mtp=True)


def test_it_never_replaces_an_existing_output(tmp_path: Path, tree: Path):
    existing = tmp_path / "taken.gguf"
    existing.write_bytes(b"not a model")

    with pytest.raises(convert_tree.ConversionRefused, match="already exists"):
        convert_tree.convert(tree, existing)
    assert existing.read_bytes() == b"not a model"


def test_a_v_head_count_that_does_not_divide_is_refused(tmp_path: Path, tree: Path):
    config = json.loads((tree / "config.json").read_text())
    config["text_config"]["linear_num_value_heads"] = 3
    (tree / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(convert_tree.ConversionRefused, match="not a multiple of linear_num_key_heads"):
        convert_tree.convert(tree, tmp_path / "o.gguf")


def test_the_cli_dry_run_writes_the_plan_and_nothing_else(tmp_path: Path, tree: Path, capsys):
    out = tmp_path / "never.gguf"
    code = convert_tree.main(["--tree", str(tree), "--out", str(out), "--with-mtp", "--dry-run"])

    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["dry_run"] is True
    assert document["source_width"] == WIDTH and document["target_width"] == ALIGNED
    assert document["padded_tensors"] == (LAYERS + 1) * 3
    assert not out.exists()


def test_the_cli_writes_a_receipt_beside_the_gguf(tmp_path: Path, tree: Path, capsys):
    out = tmp_path / "receipted.gguf"
    receipt = tmp_path / "receipt.json"
    code = convert_tree.main(["--tree", str(tree), "--out", str(out), "--receipt", str(receipt)])

    assert code == 0
    capsys.readouterr()
    document = json.loads(receipt.read_text(encoding="utf-8"))
    assert document["architecture"] == "qwen35"
    assert document["tensors"] == len(_shapes(out))
    assert document["bytes_written"] > 0


def test_a_refusal_exits_two_and_names_itself(tmp_path: Path, capsys):
    tree = _tree(tmp_path / "unsupported")
    config = _config()
    config["model_type"] = "llama"
    config.pop("text_config")
    (tree / "config.json").write_text(json.dumps(config), encoding="utf-8")

    code = convert_tree.main(["--tree", str(tree), "--out", str(tmp_path / "o.gguf")])

    assert code == 2
    assert "convert-tree:" in capsys.readouterr().err

def test_the_cli_refuses_to_overwrite_a_receipt(tmp_path: Path, tree: Path, capsys):
    receipt = tmp_path / "receipt.json"
    receipt.write_text("preserve me", encoding="utf-8")

    code = convert_tree.main(["--tree", str(tree), "--out", str(tmp_path / "model.gguf"), "--receipt", str(receipt)])

    assert code == 2
    assert receipt.read_text(encoding="utf-8") == "preserve me"


def test_the_cli_refuses_an_output_reserved_by_another_conversion(tmp_path: Path, tree: Path, capsys):
    out = tmp_path / "model.gguf"
    reservation = out.with_name(f".{out.name}.apostate-reservation")
    reservation.write_text("other conversion", encoding="utf-8")

    code = convert_tree.main(["--tree", str(tree), "--out", str(out)])

    assert code == 2
    assert not out.exists()
    assert reservation.read_text(encoding="utf-8") == "other conversion"
    assert "reserved" in capsys.readouterr().err
