class GeminiASRError(Exception):
    """Base error for GeminiASR."""


class ConfigError(GeminiASRError):
    """Configuration related errors."""


class TranscriptionError(GeminiASRError):
    """Transcription related errors."""
