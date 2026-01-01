import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/"
DEFAULT_OPENAI_COMPAT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
ALLOWED_API_SOURCES = {"gemini", "openai"}

from dotenv import load_dotenv

try:
    import tomllib
except ImportError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib


@dataclass
class TranscriptionConfig:
    duration: int = 900
    lang: str = "zh-TW"
    model: str = "gemini-2.5-flash"
    save_raw: bool = False
    skip_existing: bool = False
    preview: bool = False
    max_segment_retries: int = 3


@dataclass
class ProcessingConfig:
    max_workers: int | None = None
    ignore_keys_limit: bool = False
    timeout: int = 600


@dataclass
class LoggingConfig:
    debug: bool = False


@dataclass
class ApiConfig:
    source: str = "gemini"
    google_api_keys: list[str] = field(default_factory=list)
    base_url: str = DEFAULT_GEMINI_BASE_URL


@dataclass
class Config:
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    extra_prompt: str | None = None

    def validate(self) -> None:
        if self.transcription.duration <= 0:
            raise ValueError("duration must be positive")
        if self.processing.max_workers is not None and self.processing.max_workers <= 0:
            raise ValueError("max_workers must be positive or None")
        if self.processing.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.transcription.max_segment_retries < 0:
            raise ValueError("max_segment_retries must be >= 0")
        source = (self.api.source or "").strip().lower()
        if source not in ALLOWED_API_SOURCES:
            raise ValueError(f"api.source must be one of {sorted(ALLOWED_API_SOURCES)}")
        self.api.source = source
        if source == "openai" and self.api.base_url == DEFAULT_GEMINI_BASE_URL:
            self.api.base_url = DEFAULT_OPENAI_COMPAT_BASE_URL


def _parse_bool(value: str) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _parse_int(value: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_api_source(value: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ALLOWED_API_SOURCES:
        return normalized
    return None


class ConfigManager:
    CONFIG_SEARCH_PATHS = [
        Path("./config.toml"),
        Path("./.geminiasr/config.toml"),
        Path("~/.geminiasr/config.toml").expanduser(),
        Path("~/.config/geminiasr/config.toml").expanduser(),
    ]

    ENV_MAPPINGS: dict[str, tuple[str, str, Callable[[str], object]]] = {
        "GEMINIASR_API_SOURCE": ("api", "source", _parse_api_source),
        "GEMINIASR_LANG": ("transcription", "lang", str),
        "GEMINIASR_MODEL": ("transcription", "model", str),
        "GEMINIASR_DURATION": ("transcription", "duration", _parse_int),
        "GEMINIASR_SAVE_RAW": ("transcription", "save_raw", _parse_bool),
        "GEMINIASR_SKIP_EXISTING": ("transcription", "skip_existing", _parse_bool),
        "GEMINIASR_PREVIEW": ("transcription", "preview", _parse_bool),
        "GEMINIASR_MAX_SEGMENT_RETRIES": (
            "transcription",
            "max_segment_retries",
            _parse_int,
        ),
        "GEMINIASR_MAX_WORKERS": ("processing", "max_workers", _parse_int),
        "GEMINIASR_IGNORE_KEYS_LIMIT": ("processing", "ignore_keys_limit", _parse_bool),
        "GEMINIASR_TIMEOUT": ("processing", "timeout", _parse_int),
        "GEMINIASR_DEBUG": ("logging", "debug", _parse_bool),
        "GEMINIASR_EXTRA_PROMPT": ("extra_prompt", "", str),
        "BASE_URL": ("api", "base_url", str),
        "GEMINIASR_BASE_URL": ("api", "base_url", str),
    }

    KEY_ENV_VARS = ["GOOGLE_API_KEY"]

    @staticmethod
    def load_config(config_path: str | None = None) -> Config:
        load_dotenv()

        config = Config()
        toml_config = ConfigManager._load_toml(config_path)
        ConfigManager._apply_toml(config, toml_config)
        ConfigManager._apply_env(config)
        config.validate()
        return config

    @staticmethod
    def _load_toml(config_path: str | None) -> dict:
        if config_path:
            path = Path(config_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            return ConfigManager._read_toml(path)

        for path in ConfigManager.CONFIG_SEARCH_PATHS:
            if path.exists():
                return ConfigManager._read_toml(path)
        return {}

    @staticmethod
    def _read_toml(path: Path) -> dict:
        with path.open("rb") as handle:
            return tomllib.load(handle)

    @staticmethod
    def _apply_toml(config: Config, toml_config: dict) -> None:
        transcription = toml_config.get("transcription", {})
        config.transcription.duration = transcription.get(
            "duration", config.transcription.duration
        )
        config.transcription.lang = transcription.get("lang", config.transcription.lang)
        config.transcription.model = transcription.get("model", config.transcription.model)
        config.transcription.save_raw = transcription.get(
            "save_raw", config.transcription.save_raw
        )
        config.transcription.skip_existing = transcription.get(
            "skip_existing", config.transcription.skip_existing
        )
        config.transcription.preview = transcription.get(
            "preview", config.transcription.preview
        )
        config.transcription.max_segment_retries = transcription.get(
            "max_segment_retries", config.transcription.max_segment_retries
        )

        processing = toml_config.get("processing", {})
        config.processing.max_workers = processing.get(
            "max_workers", config.processing.max_workers
        )
        config.processing.ignore_keys_limit = processing.get(
            "ignore_keys_limit", config.processing.ignore_keys_limit
        )
        config.processing.timeout = processing.get(
            "timeout", config.processing.timeout
        )

        logging_config = toml_config.get("logging", {})
        config.logging.debug = logging_config.get("debug", config.logging.debug)

        api_config = toml_config.get("api", {})
        config.api.source = api_config.get("source", config.api.source)
        config.api.google_api_keys = api_config.get(
            "google_api_keys", config.api.google_api_keys
        )

        advanced = toml_config.get("advanced", {})
        config.extra_prompt = advanced.get("extra_prompt", config.extra_prompt)
        config.api.base_url = advanced.get("base_url", config.api.base_url)

    @staticmethod
    def _apply_env(config: Config) -> None:
        for env_var, (section, field_name, parser) in ConfigManager.ENV_MAPPINGS.items():
            raw = os.getenv(env_var)
            if raw is None or raw == "":
                continue
            parsed = parser(raw)
            if parsed is None:
                continue
            if section == "extra_prompt":
                config.extra_prompt = str(parsed)
                continue
            target = getattr(config, section)
            setattr(target, field_name, parsed)

        for key_env in ConfigManager.KEY_ENV_VARS:
            raw_keys = os.getenv(key_env)
            if raw_keys:
                keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
                if keys:
                    config.api.google_api_keys = keys
                    break

    @staticmethod
    def log_config_details(config: Config, logger) -> None:
        logger.debug("=== 配置設定詳細資訊 ===")
        logger.debug("分段時長: %s 秒", config.transcription.duration)
        logger.debug("語言設定: %s", config.transcription.lang)
        logger.debug("Gemini 模型: %s", config.transcription.model)
        logger.debug("保存原始轉錄: %s", config.transcription.save_raw)
        logger.debug("跳過已存在檔案: %s", config.transcription.skip_existing)
        logger.debug("預覽模式: %s", config.transcription.preview)
        logger.debug("偵錯模式: %s", config.logging.debug)
        logger.debug("忽略金鑰限制: %s", config.processing.ignore_keys_limit)
        logger.debug("分段最大重試次數: %s", config.transcription.max_segment_retries)
        logger.debug("API 基礎 URL: %s", config.api.base_url)
        logger.debug("API 來源: %s", config.api.source)
        logger.debug("載入的 API 金鑰數量: %s", len(config.api.google_api_keys))
        logger.debug("最大工作執行緒數: %s", config.processing.max_workers)
        logger.debug("API 逾時設定: %s 秒", config.processing.timeout)
        logger.debug("額外提示詞: %s", config.extra_prompt)
        logger.debug("========================")
