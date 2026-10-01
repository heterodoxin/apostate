from __future__ import annotations

from types import SimpleNamespace

from apostate.diode import _copy_processor_assets


def test_copy_processor_assets_restores_missing_configs_without_overwriting_saved_files(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    source_assets = {
        "processor_config.json": '{"processor_class":"Qwen3VLProcessor"}\n',
        "preprocessor_config.json": '{"image_processor_type":"Qwen2VLImageProcessor"}\n',
        "video_preprocessor_config.json": '{"video_processor_type":"Qwen3VLVideoProcessor"}\n',
    }
    for name, content in source_assets.items():
        (source / name).write_text(content, encoding="utf-8")

    saved_preprocessor = '{"image_processor_type":"saved-by-transformers"}\n'
    (output / "preprocessor_config.json").write_text(saved_preprocessor, encoding="utf-8")

    copied = _copy_processor_assets(
        str(source), SimpleNamespace(name_or_path=str(source)), str(output)
    )

    assert copied == ["processor_config.json", "video_preprocessor_config.json"]
    assert (output / "processor_config.json").read_text(encoding="utf-8") == source_assets[
        "processor_config.json"
    ]
    assert (output / "video_preprocessor_config.json").read_text(encoding="utf-8") == source_assets[
        "video_preprocessor_config.json"
    ]
    assert (output / "preprocessor_config.json").read_text(encoding="utf-8") == saved_preprocessor
