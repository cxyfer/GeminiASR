import logging
import threading

_log_lock = threading.RLock()


class _ThreadSafeHandler(logging.StreamHandler):
    def emit(self, record):
        with _log_lock:
            super().emit(record)


class _ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
        "RESET": "\033[0m",
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            color = self.COLORS[levelname]
            record.levelname = f"{color}{levelname}{self.COLORS['RESET']}"
            record.msg = f"{color}{record.msg}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(debug: bool = False) -> logging.Logger:
    logger = logging.getLogger("geminiasr")
    level = logging.DEBUG if debug else logging.INFO

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = _ThreadSafeHandler()
    formatter = _ColoredFormatter(
        "%(asctime)s - %(levelname)s - %(threadName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False

    logging.getLogger().setLevel(logging.WARNING)
    logger.debug("日誌系統已初始化，級別: %s", logging.getLevelName(level))
    return logger
