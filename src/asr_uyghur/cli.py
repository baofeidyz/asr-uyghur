"""Command line interface for Uyghur speech recognition."""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path
from threading import Thread

DEFAULT_MODEL_ID = "ixxan/whisper-small-uyghur-thugy20"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asr-uyghur",
        description="Transcribe Uyghur speech from an audio file.",
    )
    parser.add_argument("audio", type=Path, help="Audio file path")
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress logs and only output the transcription result",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write the transcription result to the specified file",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model ID or local model path (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=440,
        help="Maximum number of generated transcription tokens",
    )
    return parser


def configure_quiet_mode() -> None:
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)


def resolve_model_path(model_id: str) -> str:
    model_path = Path(model_id)
    if model_path.exists():
        return str(model_path)

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(repo_id=model_id, local_files_only=True)
    except LocalEntryNotFoundError:
        return model_id


def transcribe(
    audio_path: Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 440,
) -> str:
    import librosa
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, TextIteratorStreamer

    local_model_path = resolve_model_path(model_id)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float32

    processor = AutoProcessor.from_pretrained(local_model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        local_model_path,
        torch_dtype=torch_dtype,
    )
    model.to(device)

    sampling_rate = processor.feature_extractor.sampling_rate
    audio, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)

    inputs = processor(
        audio,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device, dtype=torch_dtype)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="uzbek",
        task="transcribe",
    )

    generation_kwargs = {
        "input_features": input_features,
        "forced_decoder_ids": forced_decoder_ids,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    collected: list[str] = []
    for text in streamer:
        print(text, end="", flush=True)
        collected.append(text)
    print()

    thread.join()
    return "".join(collected)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.quiet:
        configure_quiet_mode()
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()

    if not args.audio.exists():
        parser.error(f"audio file does not exist: {args.audio}")

    result = transcribe(
        args.audio,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
    )

    if args.output:
        args.output.write_text(result, encoding="utf-8")

    return 0
