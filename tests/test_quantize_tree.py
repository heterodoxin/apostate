"""What `apostate quantize-tree` decides, what it hands to llama-quantize, and what it refuses to guess.

The command is a chain -- convert the tree, grow the matrix, export the projector, quantize -- and the
parts worth pinning are the ones a reader cannot check by eye:

* the MTP pin: composed from the *converted file's* own draft index, and refused when it names no tensor,
  because llama-quantize matches tensor-type patterns with `std::regex_search` and reports a pattern that
  matched nothing with complete silence. Measured against the build in `D:\\AI\\loaders\\llamacpp`
  (`0.4.0-dev`, build 10845): a recipe whose only entry names no tensor exits 0, prints no warning, and
  quantizes every tensor as if the recipe had not been passed;
* the projector hybrid rule, reproduced from the tooling repository's recipe writer -- whose F32 clause is
  easy to "improve" into a projector that weighs more than the published one under the same name;
* each refusal, every one of which is a downgrade this command exists to prevent;
* the chain itself: the order of the legs, the argv each one gets, and what is deleted afterwards.

Every fixture is synthetic and tiny, so the suite needs no model, no GPU and no llama.cpp. The stub
quantizer and the stub converter stand in for the two external binaries: they record the argv they were
handed and write the artifact they were asked for, which is what makes the chain's composition observable
without a 27B checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
gguf = pytest.importorskip("gguf")
from gguf import GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from apostate import quantize_gguf, quantize_tree  # noqa: E402

HIDDEN = 8
WIDTH = 5            # the "17409": one appended neuron, deliberately not a block multiple
DRAFT_WIDTH = 4      # the "17408": the head the additive edit never touches
ALIGNED = 256
LAYERS = 1
DRAFT_INDEX = LAYERS  # the draft block starts right after the last decoder layer
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 4
VOCAB = 8
DRAFT_TENSORS = 10    # the mtp.* tensors in the fixture tree


# --- fixtures -------------------------------------------------------------------------------------------


def _bf16(*shape: int, start: float = 1.0):
    count = int(np.prod(shape))
    return (torch.arange(count, dtype=torch.float32).reshape(shape) + start).to(torch.bfloat16)


def _config(*, with_draft: bool = True, width: int = WIDTH) -> dict:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": LAYERS,
            "hidden_size": HIDDEN,
            "intermediate_size": width,
            "max_position_embeddings": 128,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "rms_norm_eps": 1e-06,
            "partial_rotary_factor": 0.5,
            # One means every layer is full attention, so the fixture needs no linear-attention tensors.
            "full_attention_interval": 1,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 2,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 2,
            "vocab_size": VOCAB,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "mtp_num_hidden_layers": 1 if with_draft else 0,
            "rope_parameters": {"rope_theta": 1000000, "mrope_section": [1, 1, 1]},
        },
    }


def _tree(
    path: Path,
    *,
    with_draft: bool = True,
    vision: bool = False,
    width: int = WIDTH,
    draft_width: int = DRAFT_WIDTH,
) -> Path:
    """A miniature bake: one decoder layer and, on request, the draft head and a vision tower."""
    path.mkdir(parents=True, exist_ok=True)
    prefix = "model.language_model.layers.0"
    tensors: dict[str, object] = {
        "model.language_model.embed_tokens.weight": _bf16(VOCAB, HIDDEN),
        "model.language_model.norm.weight": _bf16(HIDDEN),
        "lm_head.weight": _bf16(VOCAB, HIDDEN, start=50.0),
        f"{prefix}.input_layernorm.weight": _bf16(HIDDEN),
        f"{prefix}.post_attention_layernorm.weight": _bf16(HIDDEN),
        f"{prefix}.mlp.gate_proj.weight": _bf16(width, HIDDEN),
        f"{prefix}.mlp.up_proj.weight": _bf16(width, HIDDEN),
        f"{prefix}.mlp.down_proj.weight": _bf16(HIDDEN, width),
        f"{prefix}.self_attn.q_proj.weight": _bf16(HEADS * HEAD_DIM, HIDDEN),
        f"{prefix}.self_attn.k_proj.weight": _bf16(KV_HEADS * HEAD_DIM, HIDDEN),
        f"{prefix}.self_attn.v_proj.weight": _bf16(KV_HEADS * HEAD_DIM, HIDDEN),
        f"{prefix}.self_attn.o_proj.weight": _bf16(HIDDEN, HEADS * HEAD_DIM),
    }
    if vision:
        tensors["model.visual.blocks.0.attn.qkv.weight"] = _bf16(3 * HIDDEN, HIDDEN)
        tensors["model.visual.blocks.0.mlp.fc1.weight"] = _bf16(4 * HIDDEN, HIDDEN)
    if with_draft:
        tensors["mtp.fc.weight"] = _bf16(HIDDEN, 2 * HIDDEN)
        tensors["mtp.norm.weight"] = _bf16(HIDDEN)
        tensors["mtp.pre_fc_norm_embedding.weight"] = _bf16(HIDDEN)
        tensors["mtp.pre_fc_norm_hidden.weight"] = _bf16(HIDDEN)
        tensors["mtp.layers.0.input_layernorm.weight"] = _bf16(HIDDEN)
        tensors["mtp.layers.0.post_attention_layernorm.weight"] = _bf16(HIDDEN)
        tensors["mtp.layers.0.mlp.gate_proj.weight"] = _bf16(draft_width, HIDDEN)
        tensors["mtp.layers.0.mlp.up_proj.weight"] = _bf16(draft_width, HIDDEN)
        tensors["mtp.layers.0.mlp.down_proj.weight"] = _bf16(HIDDEN, draft_width)
        tensors["mtp.layers.0.self_attn.q_proj.weight"] = _bf16(HEADS * HEAD_DIM, HIDDEN)

    save_file(tensors, str(path / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    (path / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": 1},
            "weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors},
        }),
        encoding="utf-8",
    )
    (path / "config.json").write_text(
        json.dumps(_config(with_draft=with_draft, width=width)), encoding="utf-8"
    )
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


def _matrix(path: Path, width: int) -> Path:
    """A published importance matrix, written against the *base* model's width."""
    writer = GGUFWriter(str(path), "")
    writer.add_key_value("general.type", "imatrix", GGUFValueType.STRING)
    writer.add_tensor(
        "blk.0.ffn_down.weight.in_sum2", np.arange(1, width + 1, dtype=np.float32).reshape(1, width)
    )
    writer.add_tensor("blk.0.ffn_down.weight.counts", np.array([32.0], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _mtp_gguf(path: Path, *, block_count: int, nextn: int | None, blocks: int) -> Path:
    """A converted-file stand-in: what the header declares, and which blocks the file actually holds."""
    writer = GGUFWriter(str(path), "qwen35")
    writer.add_block_count(block_count)
    writer.add_embedding_length(HIDDEN)
    if nextn is not None:
        writer.add_nextn_predict_layers(nextn)
    for layer in range(blocks):
        writer.add_tensor(f"blk.{layer}.attn_norm.weight", np.ones(HIDDEN, dtype=np.float32))
        writer.add_tensor(f"blk.{layer}.ffn_down.weight", np.ones((ALIGNED, HIDDEN), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _projector_gguf(path: Path) -> Path:
    """A projector with one eligible weight, one that is not, a norm, and an F32 weight."""
    writer = GGUFWriter(str(path), "clip")
    writer.add_tensor("model.visual.fc1.weight", np.zeros((64, 64), dtype=np.float16))
    writer.add_tensor("model.visual.down.weight", np.zeros((64, 65), dtype=np.float16))
    writer.add_tensor("model.visual.norm.weight", np.zeros(64, dtype=np.float16))
    writer.add_tensor("model.visual.fc2.weight", np.zeros((64, 64), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _executable(path: Path, body: str) -> Path:
    """A stand-in for an external binary: the venv's python, so the stub can import what it needs."""
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return path


QUANTIZER_STUB = '''
"""A llama-quantize stand-in: records its argv and writes the file it was asked for."""
import json
import sys
from pathlib import Path

argv = sys.argv[1:]
if "--help" in argv:
    print("usage: llama-quantize [--tensor-type] [--tensor-type-file] model type [nthreads]")
    raise SystemExit(0)
Path(__file__ + ".argv.json").write_text(json.dumps(argv), encoding="utf-8")
Path(argv[-3]).write_bytes(b"quantized")
'''

CONVERTER_STUB = '''
"""A convert_hf_to_gguf.py stand-in: writes the projector GGUF it was asked for."""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

argv = sys.argv[1:]
outfile = Path(argv[argv.index("--outfile") + 1])
writer = GGUFWriter(str(outfile), "clip")
writer.add_tensor("model.visual.fc1.weight", np.zeros((64, 64), dtype=np.float16))
writer.add_tensor("model.visual.down.weight", np.zeros((64, 65), dtype=np.float16))
writer.add_tensor("model.visual.norm.weight", np.zeros(64, dtype=np.float16))
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
'''


@pytest.fixture
def quantizer(tmp_path: Path) -> Path:
    return _executable(tmp_path / "llama-quantize", QUANTIZER_STUB)


@pytest.fixture
def converter(tmp_path: Path) -> Path:
    """A `--llama-cpp-source` checkout, whose one interesting file is the converter."""
    checkout = tmp_path / "llama.cpp"
    checkout.mkdir()
    (checkout / "convert_hf_to_gguf.py").write_text(CONVERTER_STUB, encoding="utf-8")
    return checkout


def _recorded(quantizer: Path) -> list[str]:
    """The argv the stub quantizer was last handed."""
    return json.loads(Path(f"{quantizer}.argv.json").read_text(encoding="utf-8"))


def _printed(captured: str) -> dict:
    """The document the command printed, after the argv lines it echoes ahead of it."""
    lines = captured.splitlines()
    return json.loads("\n".join(lines[lines.index("{"):]))


def _invoke(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str]:
    code = quantize_tree.main(argv)
    return code, capsys.readouterr().err


def _base(tmp_path: Path, quantizer: Path) -> list[str]:
    """The argv every acceptance-shaped case shares: a tree, an output, a matrix and a quantizer."""
    return [
        "--tree", str(tmp_path / "tree"), "--out", str(tmp_path / "out.gguf"),
        "--quantization", "Q4_K_M",
        "--imatrix", str(_matrix(tmp_path / "published.imatrix.gguf", WIDTH)),
        "--quantizer", str(quantizer),
    ]


# --- the MTP pin ----------------------------------------------------------------------------------------


def test_mtp_pin_names_the_draft_block_from_the_files_own_header(tmp_path: Path):
    """`block_count` spells the draft two ways; the shared rule is what makes one pin right for both.

    65 blocks with one MTP layer means the draft layer is counted as a block and sits at 64; 64 blocks
    with the layer at 64 (the count left to the trunk, as a trunk-only header writes it) means the same.
    """
    counted = _mtp_gguf(tmp_path / "counted.gguf", block_count=65, nextn=1, blocks=65)
    uncounted = _mtp_gguf(tmp_path / "uncounted.gguf", block_count=64, nextn=None, blocks=65)

    assert quantize_tree.draft_pin(counted, "Q8_0") == "blk.64.*:Q8_0"
    assert quantize_tree.draft_pin(uncounted, "Q8_0") == "blk.64.*:Q8_0"


def test_mtp_pin_is_refused_when_it_names_no_tensor(tmp_path: Path):
    """The silent-failure guard: a pin that matches nothing changes nothing and says nothing.

    llama-quantize would exit 0 with the draft block at the base type, so the pin is checked against the
    converted file's own tensor names rather than against the arithmetic that produced the index.
    """
    undeclared = _mtp_gguf(tmp_path / "no-nextn.gguf", block_count=64, nextn=None, blocks=64)
    dropped = _mtp_gguf(tmp_path / "dropped.gguf", block_count=64, nextn=1, blocks=63)

    for path, index in ((undeclared, 64), (dropped, 63)):
        with pytest.raises(quantize_tree.TreeQuantizationRefused, match=rf"blk\.{index}\.\*"):
            quantize_tree.draft_pin(path, "Q8_0")


def test_mtp_kind_is_refused_when_it_is_not_a_type_name():
    assert quantize_tree.quantization_kind("q8_0") == "Q8_0"
    assert quantize_tree.quantization_kind(None) is None

    with pytest.raises(quantize_tree.TreeQuantizationRefused, match="ggml type name"):
        quantize_tree.quantization_kind("Q8 0")


# --- the projector recipe -------------------------------------------------------------------------------


def test_hybrid_recipe_is_the_tooling_repositories_rule(tmp_path: Path):
    """2-D with `ne[0]` a multiple of 32 -> Q8_0, everything else at its source type.

    `down` is the case the axis matters for: its HF shape is (64, 65), so its `ne[0]` is 65 and a rule
    that read the other axis would quantize it, producing a projector the hybrid name does not describe.
    `fc2` is the F32 clause: the reference selects only f16/bf16, and promoting F32 would make this
    projector weigh more than the published one under the identical name.
    """
    recipe = quantize_gguf.hybrid_recipe(GGUFReader(str(_projector_gguf(tmp_path / "projector.gguf"))))

    assert len(recipe) == 4
    assert dict(recipe) == {
        "model.visual.fc1.weight": "q8_0",
        "model.visual.down.weight": "f16",
        "model.visual.norm.weight": "f16",
        "model.visual.fc2.weight": "f32",
    }


def test_recipe_reaches_llama_quantize_as_a_file_or_as_flags(tmp_path: Path):
    """One mechanism or the other, and the same recipe either way."""
    recipe = [("model.visual.fc1.weight", "q8_0"), ("model.visual.norm.weight", "f16")]
    recipe_file = quantize_gguf.write_recipe(tmp_path / "recipe.txt", recipe)
    quantizer, source, out = tmp_path / "llama-quantize", tmp_path / "f16.gguf", tmp_path / "hybrid.gguf"

    assert recipe_file.read_bytes() == b"model.visual.fc1.weight=q8_0\nmodel.visual.norm.weight=f16\n"

    as_file = quantize_gguf.projector_command(
        quantizer, source, out, "Q8_0", recipe, 8, recipe_file=recipe_file
    )
    as_flags = quantize_gguf.projector_command(quantizer, source, out, "Q8_0", recipe, 8)

    assert as_file == [
        str(quantizer), "--tensor-type-file", str(recipe_file),
        str(source), str(out), "q8_0", "8",
    ]
    assert as_flags == [
        str(quantizer),
        "--tensor-type", "model.visual.fc1.weight=q8_0",
        "--tensor-type", "model.visual.norm.weight=f16",
        str(source), str(out), "q8_0", "8",
    ]


def test_supports_tensor_type_file_reads_the_binarys_help(tmp_path: Path):
    """The build in `D:\\AI\\loaders\\llamacpp` documents the flag; a build that does not, does not."""
    documented = _executable(tmp_path / "documented", 'print("usage: llama-quantize [--tensor-type-file]")\n')
    silent = _executable(tmp_path / "silent", 'print("usage: llama-quantize [--tensor-type]")\n')

    assert quantize_gguf.supports_tensor_type_file(documented) is True
    assert quantize_gguf.supports_tensor_type_file(silent) is False
    assert quantize_gguf.supports_tensor_type_file(tmp_path / "not-executable") is False


# --- what the chain refuses -----------------------------------------------------------------------------


def test_a_run_with_no_matrix_is_refused_unless_it_says_static(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """A static build shipped under an imatrix name is the downgrade this refusal exists for."""
    code, err = _invoke(capsys, [
        "--tree", str(_tree(tmp_path / "tree")), "--out", str(tmp_path / "out.gguf"),
        "--quantization", "Q4_K_M",
    ])

    assert code == 1
    assert "no importance matrix to grow" in err and "--no-imatrix" in err
    assert not (tmp_path / "out.gguf").exists()


def test_an_existing_output_is_refused_before_anything_is_converted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path
):
    _tree(tmp_path / "tree")
    out = tmp_path / "out.gguf"
    out.write_bytes(b"already here")

    code, err = _invoke(capsys, _base(tmp_path, quantizer))

    assert code == 1
    assert "refusing to overwrite output" in err
    assert out.read_bytes() == b"already here"


def test_mtp_quantization_without_a_draft_block_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path
):
    """A tree with no `mtp.*` tensors has nothing to pin, and saying so beats a silent no-op."""
    _tree(tmp_path / "tree", with_draft=False)
    code, err = _invoke(capsys, _base(tmp_path, quantizer) + ["--mtp-quantization", "Q8_0"])

    assert code == 1
    assert "holds no mtp.* tensors" in err


def test_mtp_quantization_with_no_mtp_names_the_conflict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path
):
    _tree(tmp_path / "tree")
    code, err = _invoke(
        capsys, _base(tmp_path, quantizer) + ["--mtp-quantization", "Q8_0", "--no-mtp"]
    )

    assert code == 1
    assert "--no-mtp" in err and "nothing" in err


def test_an_already_quantized_tree_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path
):
    """Re-quantizing a quantized checkpoint is a loss on top of a loss; the config says which it is."""
    tree = _tree(tmp_path / "tree")
    config = json.loads((tree / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"] = {"quant_method": "awq", "bits": 4}
    (tree / "config.json").write_text(json.dumps(config), encoding="utf-8")

    code, err = _invoke(capsys, _base(tmp_path, quantizer))

    assert code == 1
    assert "awq" in err


def test_export_mmproj_without_a_converter_names_both_places(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path, monkeypatch: pytest.MonkeyPatch
):
    """The fork bundles no llama.cpp: the refusal has to say where the converter could come from."""
    monkeypatch.delenv(quantize_gguf.LLAMA_CPP_SOURCE_VARIABLE, raising=False)
    monkeypatch.setattr(quantize_gguf.shutil, "which", lambda _name: None)
    _tree(tmp_path / "tree", vision=True)

    code, err = _invoke(capsys, _base(tmp_path, quantizer) + ["--export-mmproj"])

    assert code == 1
    assert "--llama-cpp-source" in err and "PATH" in err
    assert quantize_gguf.LLAMA_CPP_SOURCE_VARIABLE in err, "the third mechanism is the one being asked for"


def test_a_text_only_projector_source_is_refused_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path, converter: Path
):
    _tree(tmp_path / "tree")
    code, err = _invoke(
        capsys, _base(tmp_path, quantizer) + ["--export-mmproj", "--llama-cpp-source", str(converter)]
    )

    assert code == 1
    assert "holds no vision tower" in err and "model.visual." in err


def test_a_missing_projector_source_directory_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path, converter: Path
):
    _tree(tmp_path / "tree", vision=True)
    code, err = _invoke(capsys, _base(tmp_path, quantizer) + [
        "--export-mmproj", "--llama-cpp-source", str(converter),
        "--mmproj-source", str(tmp_path / "nowhere"),
    ])

    assert code == 1
    assert "not a directory" in err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--imatrix-source", "mradermacher"], "--base-model is required with --imatrix-source"),
        (["--mmproj-quantization", "Q8_0"], "require --export-mmproj"),
        (["--no-imatrix", "--imatrix", "x"], "--no-imatrix cannot be combined with --imatrix"),
    ],
)
def test_flag_combinations_that_cannot_mean_anything_are_rejected(
    tmp_path: Path, extra: list[str], message: str, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(SystemExit) as exit_info:
        quantize_tree.main([
            "--tree", str(tmp_path / "tree"), "--out", str(tmp_path / "out.gguf"),
            "--quantization", "Q4_K_M", *extra,
        ])

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


# --- the whole chain ------------------------------------------------------------------------------------


def test_dry_run_prints_the_whole_plan_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path, converter: Path
):
    """The plan is every decision the run would make, and it must leave the filesystem untouched."""
    _tree(tmp_path / "tree", vision=True)
    matrix = _matrix(tmp_path / "published.imatrix.gguf", WIDTH)
    before = sorted(path.name for path in tmp_path.rglob("*"))

    code = quantize_tree.main([
        "--tree", str(tmp_path / "tree"), "--out", str(tmp_path / "bake-Q4_K_M.gguf"),
        "--quantization", "Q4_K_M", "--imatrix", str(matrix),
        "--quantizer", str(quantizer), "--mtp-quantization", "Q8_0", "--export-mmproj",
        "--llama-cpp-source", str(converter), "--mmproj-quantization", "Q8_0", "--dry-run",
    ])
    plan = _printed(capsys.readouterr().out)

    assert code == 0
    assert sorted(path.name for path in tmp_path.rglob("*")) == before
    assert plan["dry_run"] is True
    assert plan["width"]["source"] == WIDTH and plan["width"]["target"] == ALIGNED
    assert plan["width"]["padding"] == "auto"
    assert plan["conversion"]["padding"] == "auto" and plan["conversion"]["tensors"] > 0
    assert plan["mtp"] == {
        "converted": True,
        "tensors": DRAFT_TENSORS,
        "quantization": "Q8_0",
        "draft_index": DRAFT_INDEX,
        "pin": f"blk.{DRAFT_INDEX}.*:Q8_0",
        "checked_against": "the converted file's block_count and nextn_predict_layers at run time",
    }
    assert plan["imatrix"]["source"]["publisher"] == "local"
    assert plan["imatrix"]["grown"] == "bake-Q4_K_M.imatrix.gguf"
    assert plan["mmproj"]["f16"] == "bake-mmproj-F16.gguf"
    assert plan["mmproj"]["hybrid"] == "bake-mmproj-hybrid-Q8_0-F16.gguf"
    assert plan["quantize"]["argv"][-4:] == [
        str(tmp_path / "bake-Q4_K_M.bf16.gguf"), str(tmp_path / "bake-Q4_K_M.gguf"), "q4_k_m", "0",
    ]
    assert plan["quantize"]["argv"][plan["quantize"]["argv"].index("--imatrix") + 1] == str(
        tmp_path / "bake-Q4_K_M.imatrix.gguf"
    )


def test_dry_run_reports_aligned_overwrite_tree_without_padding_or_matrix_growth(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path
):
    """An overwrite bake keeps the base width, so dry-run must not plan padding or matrix growth."""
    tree = _tree(tmp_path / "tree", width=ALIGNED, draft_width=ALIGNED)
    matrix = _matrix(tmp_path / "published.imatrix.gguf", ALIGNED)

    code = quantize_tree.main([
        "--tree", str(tree), "--out", str(tmp_path / "overwrite-Q4_K_M.gguf"),
        "--quantization", "Q4_K_M", "--imatrix", str(matrix),
        "--quantizer", str(quantizer), "--mtp-quantization", "Q8_0", "--dry-run",
    ])
    plan = _printed(capsys.readouterr().out)

    assert code == 0
    assert plan["width"] == {
        "source": ALIGNED,
        "target": ALIGNED,
        "padding": "none",
        "note": None,
    }
    assert plan["conversion"]["padding"] == "none"
    assert plan["conversion"]["padded_tensors"] == 0
    assert plan["imatrix"]["growth"] == {
        "entries_grown": 0,
        "entries_added": 0,
        "reason": "computed from the converted tensor plan",
    }


def test_dry_run_reports_matrix_growth_when_aligned_tree_gets_narrow_matrix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path
):
    """Equal model widths do not hide adaptation needed by a stale or mismatched matrix."""
    tree = _tree(tmp_path / "tree", width=ALIGNED, draft_width=ALIGNED)
    matrix = _matrix(tmp_path / "narrow.imatrix.gguf", WIDTH)

    code = quantize_tree.main([
        "--tree", str(tree), "--out", str(tmp_path / "mismatched-Q4_K_M.gguf"),
        "--quantization", "Q4_K_M", "--imatrix", str(matrix),
        "--quantizer", str(quantizer), "--mtp-quantization", "Q8_0", "--dry-run",
    ])
    plan = _printed(capsys.readouterr().out)

    assert code == 0
    assert plan["imatrix"]["growth"] == {
        "entries_grown": 1,
        "entries_added": ALIGNED - WIDTH,
        "reason": "computed from the converted tensor plan",
    }


def test_the_chain_converts_grows_quantizes_and_writes_one_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path
):
    """One call, one receipt: the pin and the grown matrix reach llama-quantize, and only the
    deliverables survive -- the BF16 trunk and the adapter matrix are this run's intermediates."""
    tree = _tree(tmp_path / "tree")
    matrix = _matrix(tmp_path / "published.imatrix.gguf", WIDTH)
    out, receipt = tmp_path / "bake-Q4_K_M.gguf", tmp_path / "receipt.json"

    code = quantize_tree.main([
        "--tree", str(tree), "--out", str(out), "--quantization", "Q4_K_M",
        "--imatrix", str(matrix), "--quantizer", str(quantizer),
        "--mtp-quantization", "Q8_0",
        "--tensor-type", f"blk.{DRAFT_INDEX}.ffn_down.weight:Q6_K", "--tensor-type", "output.weight:Q6_K",
        "--receipt", str(receipt),
    ])
    printed = _printed(capsys.readouterr().out)
    document = json.loads(receipt.read_text(encoding="utf-8"))

    assert code == 0
    assert printed == document, "the printed receipt and the written one are the same document"
    argv = _recorded(quantizer)
    pins = [argv[index + 1] for index, value in enumerate(argv) if value == "--tensor-type"]
    # llama-quantize applies the first pattern that matches a tensor name and then stops, so the narrow,
    # deliberate pin for a draft tensor is composed first and gets the say...
    assert pins[:2] == [f"blk.{DRAFT_INDEX}.ffn_down.weight=Q6_K", "output.weight=Q6_K"], (
        "an explicit pin for a tensor inside the draft block must win over the computed default"
    )
    # ...while the derived pin stays block-wide, so it still covers the block's *other* tensors: it is a
    # default, not an override of what the operator asked for by name.
    assert pins[-1] == f"blk.{DRAFT_INDEX}.*=Q8_0", "the derived pin covers the rest of the draft block"
    assert document["quantize"]["pins"] == [
        {"pin": f"blk.{DRAFT_INDEX}.ffn_down.weight:Q6_K", "origin": "--tensor-type"},
        {"pin": "output.weight:Q6_K", "origin": "--tensor-type"},
        {"pin": f"blk.{DRAFT_INDEX}.*:Q8_0", "origin": "--mtp-quantization"},
    ]
    assert document["quantize"]["quantizer"] == {
        "path": str(quantizer), "resolved_by": "--quantizer",
    }, "the receipt names the binary that ran and which mechanism chose it"
    grown = Path(argv[argv.index("--imatrix") + 1])
    assert grown.name == "bake-Q4_K_M.imatrix.gguf"
    assert argv[-2:] == ["q4_k_m", "0"]

    assert document["schema"] == "apostate.tree-quantization.v1"
    assert document["dry_run"] is False
    assert document["tree"]["path"] == "tree"
    assert document["conversion"]["tree"] == "tree"
    assert document["conversion"]["out"] == "bake-Q4_K_M.bf16.gguf"
    assert document["mtp"]["pin"] == f"blk.{DRAFT_INDEX}.*:Q8_0"
    assert document["imatrix"]["source"]["publisher"] == "local"
    assert document["imatrix"]["source"]["sha256"] == hashlib.sha256(matrix.read_bytes()).hexdigest()
    assert document["imatrix"]["growth"]["entries_grown"] == 1
    assert document["imatrix"]["grown"] == grown.name
    assert f"blk.{DRAFT_INDEX}.ffn_down.weight" in document["unweighted_draft_tensors"]
    assert document["intermediates"] == {
        "kept": False, "paths": ["bake-Q4_K_M.bf16.gguf", "bake-Q4_K_M.imatrix.gguf"],
    }
    assert document["output"] == {
        "path": "bake-Q4_K_M.gguf",
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "bytes": out.stat().st_size,
    }
    assert document["seconds"] >= 0

    assert not (tmp_path / "bake-Q4_K_M.bf16.gguf").exists(), "the trunk is an intermediate"
    assert not grown.exists(), "the grown matrix is an intermediate"
    assert out.exists() and (tree / "config.json").exists()
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")], (
        "the chain leaves no staging file, no reservation and no recipe behind"
    )


def test_keep_intermediate_keeps_the_trunk_and_the_grown_matrix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path
):
    _tree(tmp_path / "tree")
    receipt = tmp_path / "receipt.json"
    code = quantize_tree.main(
        _base(tmp_path, quantizer) + ["--keep-intermediate", "--receipt", str(receipt)]
    )
    document = json.loads(receipt.read_text(encoding="utf-8"))

    assert code == 0
    assert document["intermediates"]["kept"] is True
    assert (tmp_path / "out.bf16.gguf").exists()
    assert (tmp_path / "out.imatrix.gguf").exists()


def test_no_imatrix_is_a_static_build_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path
):
    """With `--no-imatrix` the argv must carry no matrix, and the receipt must carry none either."""
    _tree(tmp_path / "tree")
    receipt = tmp_path / "receipt.json"
    code = quantize_tree.main([
        "--tree", str(tmp_path / "tree"), "--out", str(tmp_path / "out.gguf"),
        "--quantization", "Q4_K_M", "--quantizer", str(quantizer), "--no-imatrix",
        "--keep-intermediate", "--receipt", str(receipt),
    ])
    document = json.loads(receipt.read_text(encoding="utf-8"))

    assert code == 0
    assert "--imatrix" not in _recorded(quantizer)
    assert document["imatrix"] is None
    assert document["unweighted_draft_tensors"], "no matrix means every draft MLP tensor is unweighted"
    assert not (tmp_path / "out.imatrix.gguf").exists()


def test_export_mmproj_writes_the_f16_projector_and_the_hybrid_from_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path, converter: Path
):
    """The projector names are the tooling repository's, the hybrid is quantized *from* the F16 export,
    and the F16 stops being a deliverable the moment only the hybrid was asked for."""
    tree = _tree(tmp_path / "tree", vision=True)
    out, receipt = tmp_path / "bake-VL-Q4_K_M.gguf", tmp_path / "receipt.json"
    hybrid = tmp_path / "bake-VL-mmproj-hybrid-Q8_0-F16.gguf"
    f16 = tmp_path / "bake-VL-mmproj-F16.gguf"

    code = quantize_tree.main([
        "--tree", str(tree), "--out", str(out), "--quantization", "Q4_K_M",
        "--imatrix", str(_matrix(tmp_path / "published.imatrix.gguf", WIDTH)),
        "--quantizer", str(quantizer), "--llama-cpp-source", str(converter),
        "--export-mmproj", "--mmproj-quantization", "Q8_0", "--receipt", str(receipt),
    ])
    document = json.loads(receipt.read_text(encoding="utf-8"))
    leg = document["mmproj"]

    assert code == 0
    assert leg["converter"] == {
        "path": str(converter / "convert_hf_to_gguf.py"), "resolved_by": "--llama-cpp-source",
    }
    assert leg["f16"] == f16.name and leg["hybrid"] == hybrid.name
    assert leg["recipe"] == {"mechanism": "tensor-type-file", "entries": 3, "q8_0": 1}
    assert leg["quantize_argv"][-4:] == [str(f16), str(hybrid), "q8_0", "0"]
    assert hybrid.exists() and out.exists()
    assert not f16.exists(), "only the hybrid was asked for, so the F16 export was staging"
    assert not list(tmp_path.glob(".*recipe*")), "the recipe file does not outlive the quantize"
    assert document["conversion"]["vision_tensors_skipped"] == 2


def test_export_mmproj_f16_is_the_deliverable_and_needs_no_recipe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path, converter: Path
):
    """`F16` is the export as it stands, from the projector source and not from the trunk."""
    trunk = _tree(tmp_path / "tree")
    donor = _tree(tmp_path / "donor", with_draft=False, vision=True)
    f16 = tmp_path / "bake-mmproj-F16.gguf"

    code = quantize_tree.main([
        "--tree", str(trunk), "--out", str(tmp_path / "bake-Q4_K_M.gguf"), "--quantization", "Q4_K_M",
        "--imatrix", str(_matrix(tmp_path / "published.imatrix.gguf", WIDTH)),
        "--quantizer", str(quantizer), "--llama-cpp-source", str(converter),
        "--export-mmproj", "--mmproj-source", str(donor),
    ])
    leg = _printed(capsys.readouterr().out)["mmproj"]

    assert code == 0
    assert leg["source"] == "donor"
    assert leg["f16"] == f16.name and leg["hybrid"] is None and leg["recipe"] is None
    assert "quantize_argv" not in leg
    assert f16.exists()


# --- the two llama.cpp tools: flag, variable, PATH ------------------------------------------------------


def test_the_quantizer_variable_is_honoured_without_a_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path, monkeypatch: pytest.MonkeyPatch
):
    """One variable, no flag: the machine names its llama-quantize and the receipt says which did."""
    _tree(tmp_path / "tree")
    monkeypatch.setenv(quantize_gguf.QUANTIZER_VARIABLE, str(quantizer))

    code = quantize_tree.main([
        "--tree", str(tmp_path / "tree"), "--out", str(tmp_path / "out.gguf"),
        "--quantization", "Q4_K_M",
        "--imatrix", str(_matrix(tmp_path / "published.imatrix.gguf", WIDTH)),
        "--dry-run",
    ])
    plan = _printed(capsys.readouterr().out)

    assert code == 0
    assert plan["quantize"]["quantizer"] == {
        "path": str(quantizer), "resolved_by": quantize_gguf.QUANTIZER_VARIABLE,
    }
    assert plan["quantize"]["argv"][0] == str(quantizer)


def test_the_quantizer_flag_beats_the_variable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path, monkeypatch: pytest.MonkeyPatch
):
    named = tmp_path / "named.llama-quantize"
    named.write_text("", encoding="utf-8")
    _tree(tmp_path / "tree")
    monkeypatch.setenv(quantize_gguf.QUANTIZER_VARIABLE, str(named))

    code = quantize_tree.main(_base(tmp_path, quantizer) + ["--dry-run"])
    plan = _printed(capsys.readouterr().out)

    assert code == 0
    assert plan["quantize"]["quantizer"] == {"path": str(quantizer), "resolved_by": "--quantizer"}


def test_the_converter_variable_is_honoured_and_recorded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path, converter: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """`APOSTATE_LLAMA_CPP_SOURCE` mirrors `--llama-cpp-source`, checkout and all."""
    _tree(tmp_path / "tree", vision=True)
    monkeypatch.setenv(quantize_gguf.LLAMA_CPP_SOURCE_VARIABLE, str(converter))

    code = quantize_tree.main(_base(tmp_path, quantizer) + ["--export-mmproj", "--dry-run"])
    plan = _printed(capsys.readouterr().out)

    assert code == 0
    assert plan["mmproj"]["converter"] == {
        "path": str(converter / "convert_hf_to_gguf.py"),
        "resolved_by": quantize_gguf.LLAMA_CPP_SOURCE_VARIABLE,
    }


def test_the_checkout_flag_beats_the_converter_variable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path, converter: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")
    _tree(tmp_path / "tree", vision=True)
    monkeypatch.setenv(quantize_gguf.LLAMA_CPP_SOURCE_VARIABLE, str(elsewhere))

    code = quantize_tree.main(
        _base(tmp_path, quantizer) + ["--export-mmproj", "--llama-cpp-source", str(converter), "--dry-run"]
    )
    plan = _printed(capsys.readouterr().out)

    assert code == 0
    assert plan["mmproj"]["converter"] == {
        "path": str(converter / "convert_hf_to_gguf.py"), "resolved_by": "--llama-cpp-source",
    }


def test_an_empty_converter_variable_counts_as_unset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path, converter: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Whitespace is a shell accident, not an instruction: PATH answers instead."""
    on_path = converter / "convert_hf_to_gguf.py"
    _tree(tmp_path / "tree", vision=True)
    monkeypatch.setattr(quantize_gguf.shutil, "which", lambda _name: str(on_path))
    monkeypatch.setenv(quantize_gguf.LLAMA_CPP_SOURCE_VARIABLE, "   ")

    code = quantize_tree.main(_base(tmp_path, quantizer) + ["--export-mmproj", "--dry-run"])
    plan = _printed(capsys.readouterr().out)

    assert code == 0
    assert plan["mmproj"]["converter"] == {"path": str(on_path), "resolved_by": "PATH"}


def test_a_converter_variable_that_holds_no_converter_is_refused_not_fallen_through(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path, converter: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A wrong checkout is refused by name, even though PATH holds a converter that would have run.

    The PATH answer here is the fixture's real, runnable `convert_hf_to_gguf.py`, so a fallthrough would
    have succeeded and produced a projector: the refusal is the proof it was not taken.
    """
    empty = tmp_path / "empty-checkout"
    empty.mkdir()
    _tree(tmp_path / "tree", vision=True)
    monkeypatch.setattr(
        quantize_gguf.shutil, "which", lambda _name: str(converter / "convert_hf_to_gguf.py")
    )
    monkeypatch.setenv(quantize_gguf.LLAMA_CPP_SOURCE_VARIABLE, str(empty))

    code, err = _invoke(capsys, _base(tmp_path, quantizer) + ["--export-mmproj"])

    assert code == 1
    assert quantize_gguf.LLAMA_CPP_SOURCE_VARIABLE in err, err
    assert str(empty) in err, err
    assert "--llama-cpp-source" in err and "PATH" in err, "the refusal names all three mechanisms"
    assert not (tmp_path / "out-mmproj-F16.gguf").exists()


def test_an_existing_receipt_is_never_overwritten(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantizer: Path
):
    _tree(tmp_path / "tree")
    receipt = tmp_path / "receipt.json"
    receipt.write_text("someone else's", encoding="utf-8")

    code, err = _invoke(capsys, _base(tmp_path, quantizer) + ["--receipt", str(receipt)])

    assert code == 1
    assert "receipt already exists" in err
    assert receipt.read_text(encoding="utf-8") == "someone else's"
    assert not (tmp_path / "out.gguf").exists()
