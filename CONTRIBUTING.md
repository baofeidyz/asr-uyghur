# Contributing

Thank you for taking the time to improve this project.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Before Opening a Pull Request

Please run:

```bash
ruff check .
pytest
```

## Pull Request Guidelines

- Keep changes focused on one issue or improvement.
- Include tests for behavior changes when practical.
- Update documentation when command line usage or setup steps change.
- Describe the audio/model environment used when reporting ASR quality or
  performance changes.

## Reporting Issues

When reporting a bug, include:

- Operating system and Python version.
- Installation method.
- Command that failed.
- Relevant error message or traceback.
- Whether the model was already cached or downloaded during the run.
