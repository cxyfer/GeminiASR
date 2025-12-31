import logging
import random
import threading

from .config import ConfigManager


class ApiKeyManager:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._keys: list[str] | None = None
        self._lock = threading.RLock()
        self._initialized = True

    @property
    def api_keys(self) -> list[str]:
        if self._keys is None:
            with self._lock:
                if self._keys is None:
                    self._keys = self._load_keys()
        return self._keys

    def _load_keys(self) -> list[str]:
        config = ConfigManager.load_config()
        keys = list(config.api.google_api_keys)
        if not keys:
            raise ValueError("No API keys configured")
        return keys

    def configure(self, keys: list[str] | None) -> None:
        with self._lock:
            self._keys = list(keys) if keys else []

    def get_key(self) -> str:
        with self._lock:
            if not self.api_keys:
                logging.error("All API keys are unavailable.")
                raise ValueError("All API keys are unavailable.")
            key = random.choice(self.api_keys)
            logging.debug("Providing API key (last 6 chars: ...%s).", key[-6:])
            return key

    def disable_key(self, key: str, reason: str = "unknown") -> bool:
        with self._lock:
            if key in self.api_keys:
                self.api_keys.remove(key)
                if reason == "rate_limit":
                    logging.warning(
                        "API key (last 6 chars: ...%s) hit rate limit and removed. Remaining: %s",
                        key[-6:],
                        len(self.api_keys),
                    )
                elif reason == "banned":
                    logging.error(
                        "API key (last 6 chars: ...%s) banned and removed. Remaining: %s",
                        key[-6:],
                        len(self.api_keys),
                    )
                else:
                    logging.warning(
                        "API key (last 6 chars: ...%s) disabled (%s). Remaining: %s",
                        key[-6:],
                        reason,
                        len(self.api_keys),
                    )
                return True
            return False

    def get_available_key_count(self) -> int:
        with self._lock:
            return len(self.api_keys)


key_manager = ApiKeyManager()
