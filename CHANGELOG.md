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

## [1.2.1] - 2026-01-01

### Fixed
- Fixed hanging during transcription by implementing a default 600s API timeout.
- Fixed redundant internal retry loops in transcription logic; retries are now handled by the main scheduler.
- Fixed `'VideoFileClip' object has no attribute 'subclip'` error by migrating to MoviePy v2 (`subclipped`).
- Fixed deprecated `verbose` argument usage in `write_audiofile`.

### Changed
- Upgraded `moviepy` dependency to `>=2.0.0` (tested with v2.2.1).
- Added `timeout` configuration option to `[processing]` section (default: 600 seconds).

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
