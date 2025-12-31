# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- None

### Changed
- None

### Removed
- None

## [1.2.0] - 2026-01-01

### Added
- MIT license and packaging metadata
- `geminiasr` CLI entry point with `python -m geminiasr` support
- Modular `src/` package layout for maintainability
- Standard project docs and GitHub templates
- Examples, scripts, and a minimal pytest smoke test

### Changed
- Removed langchain dependency in favor of a lightweight prompt template
- Configuration loading now uses a whitelist of environment variables and env-first precedence
- Old `gemini_asr.py` kept as a shim to the new CLI

### Removed
- Legacy `utils/` modules replaced by the `src/geminiasr` package
