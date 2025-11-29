import logging
import os
from utils.config import LOG_DIR

def get_logger(name: str):
    """Returns a logger instance with standardized settings."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid attaching multiple handlers if already configured
    if logger.hasHandlers():
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # File handler
    log_file = os.path.join(LOG_DIR, f"{name}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
