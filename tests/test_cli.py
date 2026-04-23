from pathlib import Path

from asr_uyghur.cli import DEFAULT_MODEL_ID, build_parser, resolve_model_path


def test_parser_uses_expected_defaults() -> None:
    args = build_parser().parse_args(["sample.wav"])

    assert args.audio == Path("sample.wav")
    assert args.model_id == DEFAULT_MODEL_ID
    assert args.max_new_tokens == 440
    assert args.output is None
    assert args.quiet is False


def test_parser_accepts_output_and_model_options() -> None:
    args = build_parser().parse_args(
        [
            "sample.wav",
            "--quiet",
            "--output",
            "result.txt",
            "--model-id",
            "local/model",
            "--max-new-tokens",
            "128",
        ]
    )

    assert args.quiet is True
    assert args.output == Path("result.txt")
    assert args.model_id == "local/model"
    assert args.max_new_tokens == 128


def test_resolve_model_path_accepts_existing_local_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    assert resolve_model_path(str(model_dir)) == str(model_dir)
