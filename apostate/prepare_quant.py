"""Prepare an additive-diode BF16 GGUF and imatrix for stock llama.cpp quantization.

The diode's appended neuron changes the MLP width from a quantization-friendly value such as
17408 to 17409. Stock llama.cpp can load that model, but K-quant tensors whose first dimension is
not divisible by 256 fall back to F16. A published importance matrix also retains the old width and
llama.cpp rejects that mismatch.

This command fixes both artifacts without modifying llama.cpp and without repeating HF-to-GGUF
conversion. It rewrites the existing BF16 GGUF directly to the next declared width, then grows only
the matching imatrix statistics with zeros. The model writer declares every tensor before writing
payloads, so it streams source to destination and never creates a model-sized temporary spool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from math import prod
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


from .gguf_layout import (
    MLP_AXIS,
    QK_K,
    aligned_width,
    block_count as _shared_block_count,
    declared_width as _shared_declared_width,
    draft_threshold as _shared_draft_threshold,
    is_draft,
    is_mlp_weight,
    mlp_axis,
    projection_of,
)

_MODEL_TYPES = {"F32", "F16", "BF16"}
# A published importance matrix for a 27B model is ~14 MB. A ceiling well above that still refuses a
# repository that would turn one CLI flag into unbounded download and an in-memory copy.
IMATRIX_MAX_BYTES = 1024 * 1024 * 1024


class PreparationRefused(RuntimeError):
    """The input cannot be transformed without guessing or losing weights."""


def _deps():
    try:
        import numpy as np
        from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter
    except ImportError as error:
        raise PreparationRefused(
            "GGUF preparation needs numpy and gguf-py; install llama.cpp's gguf package first"
        ) from error
    return np, GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter


def _open(path: Path):
    _np, _types, reader_type, _values, _writer = _deps()
    if not path.is_file():
        raise PreparationRefused(f"input does not exist: {path}")
    try:
        return reader_type(str(path))
    except Exception as error:
        raise PreparationRefused(f"cannot read {path} as GGUF: {error}") from error


def _contents(field: Any) -> Any:
    value = field.contents()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _fields(reader: Any) -> dict[str, Any]:
    return {name: _contents(field) for name, field in reader.fields.items() if not name.startswith("GGUF.")}


def _integer(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise PreparationRefused(f"{label} is not an integer: {value!r}") from error
    if result <= 0:
        raise PreparationRefused(f"{label} must be positive, got {result}")
    return result


def _declared_width(fields: Mapping[str, Any]) -> tuple[int, tuple[str, ...]]:
    """The shared rule, refused as this command's own error."""
    return _shared_declared_width(fields, _integer, PreparationRefused)


def _block_count(fields: Mapping[str, Any]) -> int | None:
    return _shared_block_count(fields, _integer, PreparationRefused)


def _draft_threshold(fields: Mapping[str, Any]) -> int | None:
    """The shared rule, refused as this command's own error."""
    return _shared_draft_threshold(fields, _integer, PreparationRefused)


def _tensor_ne(tensor: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def _raw_tensor(tensor: Any):
    np, types, _reader, _values, _writer = _deps()
    kind = tensor.tensor_type.name
    if kind not in _MODEL_TYPES:
        raise PreparationRefused(
            f"{tensor.name} is {kind}; prepare the BF16/F16/F32 trunk before quantizing, not a quantized GGUF"
        )
    array = np.asarray(tensor.data)
    if kind == "BF16" and array.dtype == np.uint8:
        return np.ascontiguousarray(array).view(np.uint16)
    return array


def _model_plan(reader: Any, target_width: int) -> dict[str, Any]:
    fields = _fields(reader)
    source_width, width_keys = _declared_width(fields)
    if source_width == target_width:
        raise PreparationRefused(f"model is already {source_width} wide")
    direction = "pad" if target_width > source_width else "strip"
    if direction == "pad" and target_width % QK_K:
        raise PreparationRefused(f"target width {target_width} is not a multiple of {QK_K}")
    block_count = _block_count(fields)
    plan: list[dict[str, Any]] = []
    draft: list[str] = []
    draft_from = _draft_threshold(fields)
    for tensor in reader.tensors:
        array = _raw_tensor(tensor)
        name = str(tensor.name)
        if not is_mlp_weight(name):
            continue
        # The draft (MTP) block moves with the decoder. An additive edit never touches the head, so its
        # own width is one below the trunk's -- but llama.cpp allocates *every* block, `nextn` included,
        # from `feed_forward_length`, and a draft head left behind makes the file unloadable:
        #
        #   check_tensor_dims: tensor 'blk.64.ffn_gate.weight' has wrong shape;
        #   expected 5120, 17664, got 5120, 17408
        #
        # The zeros appended there are inert (a zero gate row contributes nothing, and the down
        # projection column it feeds is zero), so the head's own weights survive untouched.
        is_draft_block = is_draft(name, draft_from)
        if is_draft_block:
            draft.append(name)
        axis = mlp_axis(name, array.shape, source_width, PreparationRefused)
        before = int(array.shape[axis])
        if before == target_width:
            continue
        plan.append({
            "tensor": name,
            "axis": axis,
            "before": before,
            "after": target_width,
            "dtype": str(array.dtype),
            "draft": is_draft_block,
        })
    if not plan:
        raise PreparationRefused("no MLP tensors needed resizing; the model is already aligned")
    suspicious = [
        str(tensor.name) for tensor in reader.tensors
        if "ffn" in str(tensor.name) and str(tensor.name).endswith(".weight")
        and not is_draft(str(tensor.name), block_count)
        and not is_mlp_weight(str(tensor.name))
        and source_width in _tensor_ne(tensor)
    ]
    if suspicious:
        raise PreparationRefused(
            "unsupported MLP tensors would leave a mixed-width model: " + ", ".join(suspicious[:8])
        )
    return {
        "source_width": source_width,
        "target_width": target_width,
        "direction": direction,
        "feed_forward_length_keys": list(width_keys),
        "tensors_resized": len(plan),
        "draft_tensors_resized": sorted(draft),
        "plan": plan,
    }


def _value_type(value: Any):
    _np, _types, _reader, kinds, _writer = _deps()
    if isinstance(value, bool):
        return kinds.BOOL
    if isinstance(value, int):
        return kinds.UINT32 if 0 <= value < 2**32 else kinds.INT64
    if isinstance(value, float):
        return kinds.FLOAT32
    if isinstance(value, str):
        return kinds.STRING
    raise PreparationRefused(f"cannot preserve metadata value {value!r} ({type(value).__name__})")


def _copy_model_metadata(writer: Any, fields: Mapping[str, Any], target_width: int, source: Path, count: int) -> None:
    if "general.alignment" in fields:
        writer.add_custom_alignment(_integer(fields["general.alignment"], "general.alignment"))
    for key, value in fields.items():
        if key in ("general.alignment", "general.architecture"):
            continue
        if key.endswith(".feed_forward_length"):
            if isinstance(value, (list, tuple)):
                writer.add_array(key, [target_width] * len(value))
            else:
                writer.add_key_value(key, target_width, _value_type(value))
        elif isinstance(value, (list, tuple)):
            writer.add_array(key, list(value))
        else:
            writer.add_key_value(key, value, _value_type(value))
    # The basename only: these artifacts exist to be quantized and published, so a full local path
    # would disclose the operator's home layout. Full paths stay in the local receipt.
    writer.add_string("apostate.quant_preparation.source", source.name)
    writer.add_uint32("apostate.quant_preparation.target_width", target_width)
    writer.add_uint32("apostate.quant_preparation.tensors", count)
    writer.add_string("apostate.quant_preparation.utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))


def _resized(array: Any, item: Mapping[str, Any]) -> tuple[Any, int]:
    np, _types, _reader, _values, _writer = _deps()
    axis = int(item["axis"])
    before, after = int(item["before"]), int(item["after"])
    if after > before:
        pads = [(0, 0)] * array.ndim
        pads[axis] = (0, after - before)
        return np.pad(array, pads, mode="constant"), 0
    removed_slice = [slice(None)] * array.ndim
    removed_slice[axis] = slice(after, before)
    removed = array[tuple(removed_slice)]
    if np.count_nonzero(removed):
        raise PreparationRefused(f"{item['tensor']} has non-zero weights in the region that would be removed")
    kept_slice = [slice(None)] * array.ndim
    kept_slice[axis] = slice(0, after)
    return np.ascontiguousarray(array[tuple(kept_slice)]), int(removed.nbytes)


def _entry_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _new_stage(final: Path) -> Path:
    """Reserve the final path, then return a staged sibling path to write into.

    The reservation is an exclusive create, so an existing file -- or a symlink, which is never
    followed because ``O_EXCL`` fails on one -- is refused before any work happens. Publishing is
    then ``os.replace`` of our own reservation, which is atomic on Windows and POSIX and needs no
    hard-link support: ``os.link`` would fail on exFAT and rejects ``follow_symlinks=False`` on
    Windows.
    """
    parent = final.parent
    if not parent.is_dir():
        raise PreparationRefused(f"output directory does not exist: {parent}")
    try:
        os.close(os.open(final, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
    except FileExistsError as error:
        # A zero-length file is almost always our own reservation from a killed run. Saying so turns
        # a confusing permanent refusal into an obvious one-line fix.
        interrupted = final.is_file() and final.stat().st_size == 0
        detail = "a previous run was interrupted; delete it to retry" if interrupted else "refusing to replace it"
        raise PreparationRefused(f"output already exists: {final} ({detail})") from error
    except OSError as error:
        raise PreparationRefused(f"cannot create output {final}: {error}") from error
    try:
        directory = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=parent))
    except OSError as error:
        # The reservation is already on disk, so it must not outlive a staging failure.
        final.unlink(missing_ok=True)
        raise PreparationRefused(f"cannot stage output beside {final}: {error}") from error
    return directory / final.name


def _discard_stage(stage: Path, final: Path) -> None:
    """Drop the staged file, its private directory, and the reservation we created."""
    stage.unlink(missing_ok=True)
    try:
        stage.parent.rmdir()
    except FileNotFoundError:
        pass
    final.unlink(missing_ok=True)


def _publish_stage(stage: Path, final: Path) -> None:
    try:
        os.replace(stage, final)
    finally:
        stage.unlink(missing_ok=True)
        try:
            stage.parent.rmdir()
        except FileNotFoundError:
            pass


def prepare_model(
    source: Path | str,
    out: Path | str,
    target_width: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    source, out = Path(source), Path(out)
    if source.resolve() == out.resolve():
        raise PreparationRefused("the model rewrite never writes over its input")
    if _entry_exists(out):
        raise PreparationRefused(f"output already exists: {out}")
    reader = _open(source)
    plan = _model_plan(reader, target_width)
    receipt = {"source": str(source), "out": str(out), **plan}
    if dry_run:
        return receipt

    np, _types, _reader, _values, writer_type = _deps()
    by_name = {item["tensor"]: item for item in plan["plan"]}
    architecture = str(_fields(reader).get("general.architecture", ""))
    stage = _new_stage(out)
    writer = None
    stripped = 0
    try:
        writer = writer_type(str(stage), architecture)
        _copy_model_metadata(writer, _fields(reader), target_width, source, len(by_name))
        for tensor in reader.tensors:
            array = _raw_tensor(tensor)
            item = by_name.get(str(tensor.name))
            shape = list(array.shape)
            nbytes = int(array.nbytes)
            if item is not None:
                shape[int(item["axis"])] = target_width
                nbytes = prod(shape) * int(array.dtype.itemsize)
            writer.add_tensor_info(
                str(tensor.name), shape, array.dtype, nbytes, raw_dtype=tensor.tensor_type
            )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for tensor in reader.tensors:
            array = _raw_tensor(tensor)
            item = by_name.get(str(tensor.name))
            if item is not None:
                array, removed = _resized(array, item)
                stripped += removed
            writer.write_tensor_data(array)
        writer.close()
        _publish_stage(stage, out)
    except BaseException:
        try:
            if writer is not None:
                writer.close()
        finally:
            _discard_stage(stage, out)
        raise
    receipt["stripped_bytes_verified_zero"] = stripped
    receipt["out_bytes"] = out.stat().st_size
    return receipt


def _publisher_repositories(publisher: str, base_model: str) -> list[str]:
    coordinates = base_model.strip().strip("/")
    segment = r"[A-Za-z0-9][A-Za-z0-9._-]*"
    if re.fullmatch(rf"{segment}(/{segment})?", coordinates) is None:
        raise PreparationRefused(
            f"--base-model must be a valid owner/model or model name, got {base_model!r}"
        )
    model = coordinates.split("/")[-1]
    if publisher == "mradermacher":
        return [f"mradermacher/{model}-i1-GGUF"]
    if publisher == "bartowski":
        owner_model = coordinates.replace("/", "_")
        return list(dict.fromkeys([
            f"bartowski/{model}-GGUF",
            f"bartowski/{owner_model}-GGUF",
        ]))
    raise PreparationRefused(f"unknown imatrix publisher {publisher!r}")


def _cached_imatrices(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.rglob("*")
        # A symlink is skipped deliberately: the cache must describe what it holds, and following
        # one would let a planted link be reported under the publisher's repository name.
        if not path.is_symlink()
        and path.is_file()
        and "imatrix" in path.name.lower()
        and (path.name.lower().endswith(".gguf") or path.name.lower().endswith(".imatrix"))
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_remote_name(filename: str) -> str:
    """A repository-relative path that cannot escape the cache directory it is written into."""
    candidate = PurePosixPath(filename)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.name:
        raise PreparationRefused(f"unsafe remote filename published by the repository: {filename!r}")
    return str(candidate)


def resolve_published_imatrix(
    publisher: str,
    base_model: str,
    cache_root: Path | str,
    *,
    api: Any = None,
    downloader: Any = None,
) -> tuple[Path, dict[str, str]]:
    """Discover one publisher's matrix, cache it, and return its exact Hub provenance."""
    repositories = _publisher_repositories(publisher, base_model)
    root = Path(cache_root).expanduser()
    for repository in repositories:
        cache = root / publisher / repository.split("/", 1)[1]
        local = _cached_imatrices(cache)
        if len(local) == 1:
            return local[0].resolve(), {
                "publisher": publisher,
                "repository": repository,
                "filename": local[0].relative_to(cache).as_posix(),
                # A cached file cannot prove which commit produced it, so the receipt says so
                # instead of implying an audit that never happened.
                "revision": None,
                "sha256": _sha256(local[0]),
                "origin": "cache",
            }
        if len(local) > 1:
            raise PreparationRefused(
                f"{cache} holds more than one imatrix; pass --imatrix to select one"
            )

    if api is None or downloader is None:
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as error:
            raise PreparationRefused(
                "automatic imatrix download needs huggingface_hub; install .[gguf] or pass --imatrix"
            ) from error
        api = api or HfApi()
        downloader = downloader or hf_hub_download

    failures: list[str] = []
    for repository in repositories:
        try:
            info = api.repo_info(repository, files_metadata=True)
            candidates = {
                sibling.rfilename: getattr(sibling, "size", None)
                for sibling in (getattr(info, "siblings", None) or [])
                if "imatrix" in PurePosixPath(sibling.rfilename).name.lower()
                and sibling.rfilename.lower().endswith((".gguf", ".imatrix"))
            }
        except Exception as error:
            failures.append(f"{repository}: {type(error).__name__}")
            continue
        if len(candidates) != 1:
            reason = "no imatrix" if not candidates else f"{len(candidates)} imatrices"
            failures.append(f"{repository}: {reason}")
            continue
        name, size = next(iter(candidates.items()))
        filename = _safe_remote_name(name)
        if size is not None and int(size) > IMATRIX_MAX_BYTES:
            raise PreparationRefused(
                f"{repository}/{filename} is too large for an importance matrix "
                f"({int(size)} bytes > {IMATRIX_MAX_BYTES}); pass --imatrix if this is genuinely wanted"
            )
        # Pin the commit the listing came from, so the receipt names a fixed artifact rather than
        # whatever the branch happens to hold later.
        revision = getattr(info, "sha", None)
        cache = root / publisher / repository.split("/", 1)[1]
        downloaded = Path(
            downloader(repo_id=repository, filename=filename, revision=revision, local_dir=str(cache))
        )
        resolved = downloaded.resolve()
        if not resolved.is_relative_to(cache.resolve()):
            raise PreparationRefused(f"download landed outside the cache directory: {resolved}")
        return resolved, {
            "publisher": publisher,
            "repository": repository,
            "filename": filename,
            "revision": revision,
            "sha256": _sha256(resolved),
            "origin": "hub",
        }
    raise PreparationRefused(
        f"no {publisher} imatrix found for {base_model!r} ({'; '.join(failures)}); pass --imatrix"
    )




def _matrix_data(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    np, _types, _reader, _values, _writer = _deps()
    reader = _open(path)
    fields = _fields(reader)
    if fields.get("general.type") != "imatrix":
        raise PreparationRefused(f"{path} is not an imatrix GGUF (general.type={fields.get('general.type')!r})")
    tensors = {str(tensor.name): np.asarray(tensor.data).copy() for tensor in reader.tensors}
    if not any(name.endswith(".in_sum2") for name in tensors):
        raise PreparationRefused(f"{path} has no imatrix statistics")
    return fields, tensors


def _target_layouts(path: Path) -> dict[str, tuple[int, int]]:
    reader = _open(path)
    result: dict[str, tuple[int, int]] = {}
    for tensor in reader.tensors:
        ne = _tensor_ne(tensor)
        result[str(tensor.name)] = (ne[0], ne[2] if len(ne) > 2 else 1)
    return result


def _matrix_plan(
    matrix_tensors: Mapping[str, Any],
    target_layouts: Mapping[str, tuple[int, int]],
    expect_append: int | None,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for statistic, values in matrix_tensors.items():
        if not statistic.endswith(".in_sum2"):
            continue
        tensor = statistic.removesuffix(".in_sum2")
        if tensor not in target_layouts:
            raise PreparationRefused(f"target model has no tensor {tensor!r} named by the imatrix")
        ne0, ne2 = target_layouts[tensor]
        if values.ndim == 1:
            before, after, axis = int(values.size), ne0 * ne2, 0
        elif values.ndim == 2 and int(values.shape[0]) == ne2:
            before, after, axis = int(values.shape[1]), ne0, 1
        else:
            raise PreparationRefused(
                f"imatrix entry {tensor} has unsupported shape {tuple(values.shape)} for target ne2={ne2}"
            )
        if after < before:
            raise PreparationRefused(f"imatrix entry {tensor} is wider than the target model ({before} > {after})")
        if after > before:
            if ne0 % QK_K:
                raise PreparationRefused(f"target width {ne0} for {tensor} is not a multiple of {QK_K}")
            added = (after - before) * (ne2 if values.ndim == 2 else 1)
            if expect_append is not None and added != expect_append:
                raise PreparationRefused(
                    f"{tensor} grows by {added}, which does not equal --expect-append {expect_append}"
                )
            plan.append({
                "tensor": tensor,
                "statistic": statistic,
                "axis": axis,
                "before": before,
                "after": after,
                "added": added,
            })
    return plan


def _copy_matrix_metadata(writer: Any, fields: Mapping[str, Any]) -> None:
    for key, value in fields.items():
        if key == "general.architecture":
            continue
        if isinstance(value, (list, tuple)):
            writer.add_array(key, list(value))
        else:
            writer.add_key_value(key, value, _value_type(value))


def _unweighted_draft_mlp_tensors(target: Path, matrix_tensors: Mapping[str, Any]) -> list[str]:
    """Draft MLP weights missing imatrix statistics, which llama.cpp would quantize unweighted."""
    reader = _open(target)
    threshold = _draft_threshold(_fields(reader))
    statistics = {
        name.removesuffix(".in_sum2") for name in matrix_tensors if name.endswith(".in_sum2")
    }
    return sorted(
        str(tensor.name)
        for tensor in reader.tensors
        if is_draft(str(tensor.name), threshold)
        and is_mlp_weight(str(tensor.name))
        and str(tensor.name) not in statistics
    )


def adapt_matrix(
    imatrix: Path | str,
    target: Path | str,
    out: Path | str,
    *,
    expect_append: int | None = None,
    allow_unweighted: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    imatrix, target, out = Path(imatrix), Path(target), Path(out)
    if imatrix.resolve() == out.resolve():
        raise PreparationRefused("the imatrix rewrite never writes over its source")
    if _entry_exists(out):
        raise PreparationRefused(f"output already exists: {out}")
    fields, tensors = _matrix_data(imatrix)
    unweighted_draft = _unweighted_draft_mlp_tensors(target, tensors)
    if unweighted_draft and not allow_unweighted:
        raise PreparationRefused(
            f"{len(unweighted_draft)} draft MLP tensor(s) have no imatrix statistics: "
            f"{', '.join(unweighted_draft[:4])}. llama.cpp would quantize them unweighted; "
            "pass --allow-unweighted to accept that explicitly"
        )
    plan = _matrix_plan(tensors, _target_layouts(target), expect_append)
    receipt: dict[str, Any] = {
        "source": str(imatrix),
        "target": str(target),
        "out": str(out),
        "entries_total": sum(name.endswith(".in_sum2") for name in tensors),
        "entries_grown": len(plan),
        "entries_added": sum(item["added"] for item in plan),
        "plan": plan,
        "unweighted_draft_tensors": unweighted_draft,
    }
    if not plan:
        receipt["reason"] = "imatrix already matches the target; output is a provenance copy"
    if dry_run:
        return receipt

    np, _types, _reader, _values, writer_type = _deps()
    by_name = {item["statistic"]: item for item in plan}
    stage = _new_stage(out)
    writer = None
    try:
        writer = writer_type(str(stage), "")
        _copy_matrix_metadata(writer, fields)
        writer.add_string("apostate.quant_preparation.imatrix_from", imatrix.name)
        writer.add_string("apostate.quant_preparation.target", target.name)
        writer.add_string("apostate.quant_preparation.appended_value", "zero")
        for name, values in tensors.items():
            item = by_name.get(name)
            if item is not None:
                padding = [(0, 0)] * values.ndim
                padding[item["axis"]] = (0, item["after"] - item["before"])
                values = np.pad(values, padding, mode="constant")
            writer.add_tensor(name, values)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        _publish_stage(stage, out)
    except BaseException:
        try:
            if writer is not None:
                writer.close()
        finally:
            _discard_stage(stage, out)
        raise

    _ignored_fields, written = _matrix_data(out)
    remaining = _matrix_plan(written, _target_layouts(target), None)
    if remaining:
        out.unlink(missing_ok=True)
        raise PreparationRefused("adapted imatrix still does not match the target model")
    receipt["out_bytes"] = out.stat().st_size
    return receipt


def _planned_target_layouts(
    reader: Any, model_plan: Mapping[str, Any]
) -> dict[str, tuple[int, int]]:
    """Imatrix layouts after applying a model plan, without creating the target GGUF."""
    layouts: dict[str, tuple[int, int]] = {}
    resized = {item["tensor"]: item for item in model_plan["plan"]}
    for tensor in reader.tensors:
        name = str(tensor.name)
        ne = _tensor_ne(tensor)
        ne0 = ne[0]
        ne2 = ne[2] if len(ne) > 2 else 1
        item = resized.get(name)
        # The intermediate dimension sits on `ne[0]` for a projection whose pinned array axis is the
        # last one (`ffn_down`), and on `ne[1]` for the gated pair -- where the matrix entry's own width
        # already carries the change. Reading that from the shared table rather than from a hardcoded
        # name keeps this in step with what the repair padded.
        if item is not None and MLP_AXIS.get(projection_of(name)) == len(ne) - 1:
            ne0 = int(item["after"])
        layouts[name] = (ne0, ne2)
    return layouts


def _matrix_preview(
    imatrix: Path,
    out: Path,
    target: Path,
    target_layouts: Mapping[str, tuple[int, int]],
    expect_append: int | None,
) -> dict[str, Any]:
    if imatrix.resolve() == out.resolve():
        raise PreparationRefused("the imatrix rewrite never writes over its source")
    if _entry_exists(out):
        raise PreparationRefused(f"output already exists: {out}")
    _fields_value, tensors = _matrix_data(imatrix)
    plan = _matrix_plan(tensors, target_layouts, expect_append)
    preview: dict[str, Any] = {
        "source": str(imatrix),
        "target": str(target),
        "out": str(out),
        "entries_total": sum(name.endswith(".in_sum2") for name in tensors),
        "entries_grown": len(plan),
        "entries_added": sum(item["added"] for item in plan),
        "plan": plan,
    }
    if not plan:
        preview["reason"] = "imatrix already matches the target"
    return preview


def _check_receipt_path(receipt: Path, args: Any, resolved_imatrix: Path | None) -> None:
    """A receipt is metadata: it may never land on an input or on a produced artifact."""
    candidates = [args.model, args.out_model, args.out_imatrix, resolved_imatrix]
    for other in candidates:
        if other is not None and Path(other).expanduser().resolve() == receipt.expanduser().resolve():
            raise PreparationRefused(f"receipt path collides with {other}")
    if _entry_exists(receipt):
        raise PreparationRefused(f"receipt already exists: {receipt}")


def _write_receipt(path: Path, document: Mapping[str, Any]) -> None:
    """Publish the receipt without replacing anything that already exists at its path."""
    if _entry_exists(path):
        raise PreparationRefused(f"receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = _new_stage(path)
    try:
        stage.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _publish_stage(stage, path)
    except BaseException:
        _discard_stage(stage, path)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apostate prepare-quant",
        description="Align an additive-diode BF16 GGUF and its imatrix for stock llama.cpp.",
    )
    parser.add_argument("--model", type=Path, required=True, help="unaligned BF16/F16/F32 GGUF")
    parser.add_argument(
        "--out-model",
        type=Path,
        help="aligned GGUF to create. Omit it when the model needs no repair -- a tree converted by "
             "`convert-tree` is already padded -- and only its imatrix has to be grown, which is the "
             "common case for a bake",
    )
    parser.add_argument(
        "--to",
        type=int,
        help="target MLP width; omit it and the width is computed as the next multiple of 256 above the "
             "model's own declared width, which is what an additive bake's `base + 1` needs",
    )
    matrix_source = parser.add_mutually_exclusive_group()
    matrix_source.add_argument("--imatrix", type=Path, help="local published imatrix to adapt")
    matrix_source.add_argument(
        "--imatrix-source",
        choices=("mradermacher", "bartowski"),
        help="discover and cache the base model's published imatrix",
    )
    parser.add_argument("--base-model", help="owner/model coordinates used with --imatrix-source")
    parser.add_argument(
        "--imatrix-cache",
        type=Path,
        default=Path("~/.cache/apostate/imatrix"),
        help="published imatrix cache (default: ~/.cache/apostate/imatrix)",
    )
    parser.add_argument("--out-imatrix", type=Path, help="adapted imatrix to create")
    parser.add_argument("--expect-append", type=int, help="exact statistic growth; catches a wrong base matrix")
    parser.add_argument(
        "--allow-unweighted",
        action="store_true",
        help="accept draft MLP tensors absent from the matrix being quantized unweighted",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and print the model plan without writing")
    parser.add_argument("--receipt", type=Path, help="write the combined JSON receipt here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    has_matrix = args.imatrix is not None or args.imatrix_source is not None
    if has_matrix != (args.out_imatrix is not None):
        parser.error("--out-imatrix is required with --imatrix or --imatrix-source")
    if args.imatrix_source is not None and not args.base_model:
        parser.error("--base-model is required with --imatrix-source")
    try:
        source_reader = _open(args.model)
        source_width, _width_keys = _declared_width(_fields(source_reader))
        target_width = args.to if args.to is not None else aligned_width(source_width)
        # Two accepted shapes, and each refusal names the flag that resolves it. A model the *tree
        # conversion* already padded needs no repair: its MLP is at a block multiple and the only thing
        # left to grow is the published matrix, which is why `--out-model` is optional. Requiring it
        # there would be the reason an operator re-converts a 27B tree with `--no-pad` purely to hand
        # this command something broken to fix.
        already_aligned = source_width == target_width
        if already_aligned and args.out_model is not None:
            raise PreparationRefused(
                f"{args.model} already declares {source_width}, which is a multiple of {QK_K}; there is "
                "no width to repair. Drop --out-model to grow the imatrix against it, or pass --to to ask "
                "for a different width"
            )
        if not already_aligned and args.out_model is None:
            raise PreparationRefused(
                f"{args.model} declares {source_width}, which is not a multiple of {QK_K}: the MLP has to "
                f"be padded to {target_width} first, so --out-model is required. (A tree converted by "
                "`convert-tree` is padded in the same pass and needs none of this.)"
            )
        if already_aligned and not has_matrix:
            raise PreparationRefused(
                "the model needs no repair and no imatrix was requested, so there is nothing to do; pass "
                "--imatrix or --imatrix-source to grow a matrix, or --to to pad further"
            )
        model_plan = None if already_aligned else _model_plan(source_reader, target_width)
        target_model = args.out_model or args.model
        if model_plan is not None and args.model.resolve() == target_model.resolve():
            raise PreparationRefused("the model rewrite never writes over its input")
        if args.out_model is not None and _entry_exists(args.out_model):
            raise PreparationRefused(f"output already exists: {args.out_model}")
        model_preview = (
            {"source": str(args.model), "out": str(args.out_model), **model_plan}
            if model_plan is not None
            else None
        )
        resolved_imatrix = args.imatrix
        source_provenance = None
        if args.imatrix_source is not None:
            resolved_imatrix, source_provenance = resolve_published_imatrix(
                args.imatrix_source, args.base_model, args.imatrix_cache
            )
        elif args.imatrix is not None:
            source_provenance = {"publisher": "local", "path": str(args.imatrix)}
        matrix_preview = None
        if resolved_imatrix is not None:
            matrix_preview = _matrix_preview(
                resolved_imatrix,
                args.out_imatrix,
                target_model,
                # With no repair the target's layout is the model's own, so the plan is empty and every
                # statistic is checked against the widths the file already declares.
                _planned_target_layouts(source_reader, model_plan or {"plan": []}),
                args.expect_append,
            )
        if args.receipt is not None:
            _check_receipt_path(args.receipt, args, resolved_imatrix)

        created: list[Path] = []
        if args.dry_run:
            model, matrix = model_preview, matrix_preview
        else:
            try:
                model = None
                if model_plan is not None:
                    model = prepare_model(args.model, args.out_model, args.to)
                    created.append(args.out_model)
                matrix = None
                if resolved_imatrix is not None:
                    matrix = adapt_matrix(
                        resolved_imatrix,
                        target_model,
                        args.out_imatrix,
                        expect_append=args.expect_append,
                        allow_unweighted=args.allow_unweighted,
                    )
                    created.append(args.out_imatrix)
            except BaseException:
                # A half-prepared pair is worse than none: the no-overwrite rule would then block
                # the identical retry that would have produced both artifacts.
                for path in created:
                    path.unlink(missing_ok=True)
                raise
        document = {
            "schema": "apostate.quant-preparation.v1",
            "dry_run": args.dry_run,
            "model": model,
            "imatrix": matrix,
            "imatrix_source": source_provenance,
        }
        if args.receipt is not None and not args.dry_run:
            try:
                _write_receipt(args.receipt, document)
            except BaseException:
                for path in created:
                    path.unlink(missing_ok=True)
                raise
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0
    except PreparationRefused as error:
        print(f"prepare-quant: refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
