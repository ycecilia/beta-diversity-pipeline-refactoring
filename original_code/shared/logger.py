import os
import sys
import logging
from datetime import datetime

# Add necessary paths to sys.path to ensure imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure global logger
logger = logging.getLogger("edna-explorer")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Check if we're in development mode
DEV_MODE = os.getenv("DEV") != "0"


def log(message, level="info"):
    """
    Log a message using the appropriate method based on DEV environment variable.
    In development mode (DEV=1), this function will use print() for immediate console visibility.
    In production mode (DEV=0), this function will use the logging module.
    The message will be captured by the LogCapture mechanism in both cases.

    Args:
        message (str): The message to log
        level (str): The log level (debug, info, warning, error, critical)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # In development mode, use print() for immediate visibility
    if DEV_MODE:
        level_prefix = {
            "debug": "[DEBUG]",
            "info": "[INFO]",
            "warning": "[WARNING]",
            "error": "[ERROR]",
            "critical": "[CRITICAL]",
        }.get(level.lower(), "[INFO]")

        print(f"{timestamp} {level_prefix} {message}")

    # In production mode, use the logger
    else:
        log_method = {
            "debug": logger.debug,
            "info": logger.info,
            "warning": logger.warning,
            "error": logger.error,
            "critical": logger.critical,
        }.get(level.lower(), logger.info)

        log_method(message)


def debug(message):
    """Log a debug message"""
    log(message, "debug")


def info(message):
    """Log an info message"""
    log(message, "info")


def warning(message):
    """Log a warning message"""
    log(message, "warning")


def error(message):
    """Log an error message"""
    log(message, "error")


def critical(message):
    """Log a critical message"""
    log(message, "critical")
