import os
import random
import threading
import logging

class ApiKeyManager:
    """
    A thread-safe API key manager that supports loading keys from various sources.
    It uses a singleton pattern to ensure a single instance manages the key state
    throughout the application.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, source='env', config_path=None):
        """
        Initializes the ApiKeyManager.

        Args:
            source (str): The source of the keys, can be 'env' (environment variables) or 'config' (config file).
            config_path (str, optional): The path to the config file if the source is 'config'.
        """
        # Prevent re-initialization
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._api_keys = []
        self._lock = threading.RLock()
        self._initialized = True

        if source == 'env':
            self._load_from_env()
        elif source == 'config' and config_path:
            self._load_from_config(config_path)
        else:
            logging.warning("No valid API KEY source specified, the key pool is empty.")

        if not self._api_keys:
            logging.error("Failed to load any API KEYs.")
            raise ValueError("No valid GOOGLE_API_KEY could be loaded.")
            
        logging.info(f"ApiKeyManager initialized, successfully loaded {len(self._api_keys)} API KEYs.")

    def _load_from_env(self):
        """Loads keys from the GOOGLE_API_KEY environment variable."""
        keys_str = os.getenv('GOOGLE_API_KEY', '')
        if keys_str:
            self._api_keys = [key.strip() for key in keys_str.split(',') if key.strip()]
            logging.debug(f"Successfully loaded {len(self._api_keys)} keys from environment variable.")
        else:
            logging.warning("Environment variable 'GOOGLE_API_KEY' is not set or is empty.")

    def _load_from_config(self, path):
        """
        (For future expansion) Loads keys from a configuration file.
        Expects one key per line in the config file.
        """
        raise NotImplementedError("Loading keys from config file is not implemented yet.")
        # logging.debug(f"Attempting to load keys from config file: '{path}'...")

    def get_key(self):
        """
        Selects a random key from the available key pool.

        Returns:
            str: An available API key.

        Raises:
            ValueError: If no keys are available.
        """
        with self._lock:
            if not self._api_keys:
                logging.error("All API KEYs have exceeded their usage limit or are unavailable.")
                raise ValueError("All API KEYs have exceeded their usage limit or are unavailable.")
            key = random.choice(self._api_keys)
            logging.debug(f"Providing API KEY (last 6 chars: ...{key[-6:]}).")
            return key

    def disable_key(self, key):
        """
        Removes a key from the pool, e.g., when it's exhausted or invalid.

        Args:
            key (str): The key to remove.

        Returns:
            bool: True if the key was successfully removed, False otherwise.
        """
        with self._lock:
            if key in self._api_keys:
                self._api_keys.remove(key)
                logging.warning(
                    f"API KEY (last 6 chars: ...{key[-6:]}) has been marked as unusable and removed from the pool. "
                    f"Remaining keys: {len(self._api_keys)}"
                )
                return True
            return False

    def get_available_key_count(self):
        """
        Returns the number of currently available API keys.

        Returns:
            int: The number of available keys.
        """
        with self._lock:
            return len(self._api_keys)

# By creating a module-level instance, we ensure that any part of the application
# that imports this module will use the *same* instance of ApiKeyManager.
# This is crucial for maintaining a consistent state of the API key pool (e.g.,
# which keys have been disabled) across different threads and modules.
# It acts as a practical singleton, making it easy to access without
# needing to pass the instance around.
key_manager = ApiKeyManager()