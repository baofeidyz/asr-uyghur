# asr-uyghur

Command line Uyghur automatic speech recognition powered by the
[`ixxan/whisper-small-uyghur-thugy20`](https://huggingface.co/ixxan/whisper-small-uyghur-thugy20)
Whisper model.

## Features

- Transcribes a local audio file from the terminal.
- Streams text while the model is generating.
- Writes the final transcription to a file when requested.
- Uses Apple Silicon MPS when available, otherwise falls back to CPU.
- Works with the default Hugging Face model or a custom local/model ID.

## Requirements

- Python 3.10 or newer.
- An environment supported by PyTorch.
- Internet access on first run to download the model from Hugging Face, unless the
  model is already cached locally.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/baofeidyz/asr-uyghur.git
cd asr-uyghur
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

## Usage

Transcribe an audio file:

```bash
asr-uyghur path/to/audio.wav
```

Write the result to a file:

```bash
asr-uyghur path/to/audio.wav --output transcript.txt
```

Suppress most dependency logs:

```bash
asr-uyghur path/to/audio.wav --quiet
```

Use a custom model ID or local model directory:

```bash
asr-uyghur path/to/audio.wav --model-id /path/to/local/model
```

The legacy script entry point is also kept:

```bash
python asr-uyghur.py path/to/audio.wav
```

## Development

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

## Project Status

This project is in alpha. The command line interface is usable, but model
quality, runtime performance, and hardware compatibility may vary by system and
audio input.

## Contributing

Issues and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before proposing changes.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
