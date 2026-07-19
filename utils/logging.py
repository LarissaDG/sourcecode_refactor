import logging
import os
from pathlib import Path


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """Cria um logger que escreve em <log_dir>/<name>.log e no console."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
