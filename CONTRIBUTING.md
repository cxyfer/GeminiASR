# Contributing

Thanks for your interest in GeminiASR! This guide keeps contributions simple and consistent.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Development Notes

- Target Python: 3.10+
- Configuration: use `config.example.toml` as a template
- Avoid committing secrets: use `.env` or untracked `config.toml`

## Lint & Test

```bash
ruff check .
pytest
```

## Pull Request Checklist

- [ ] Code is formatted and linted
- [ ] Tests added or updated when applicable
- [ ] README or docs updated if behavior changes
- [ ] No secrets or credentials included
