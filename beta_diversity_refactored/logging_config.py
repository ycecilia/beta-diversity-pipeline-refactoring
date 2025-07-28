"""
Logging configuration and utilities for beta diversity analysis.
"""

import logging
import time
import psutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from functools import wraps

from .config import get_config


@dataclass
class PerformanceMetrics:
    """Performance metrics data class."""

    start_time: float
    end_time: float
    duration: float
    memory_start: float
    memory_end: float
    memory_peak: float
    cpu_percent: Optional[float] = None


class PerformanceLogger:
    """Performance logging utility."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._metrics: Dict[str, PerformanceMetrics] = {}

    @contextmanager
    def track_operation(self, operation_name: str, log_level: int = logging.INFO):
        """Context manager to track operation performance."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        self.logger.log(log_level, f"Starting operation: {operation_name}")

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            duration = end_time - start_time

            metrics = PerformanceMetrics(
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                memory_start=start_memory,
                memory_end=end_memory,
                memory_peak=max(start_memory, end_memory),
            )

            self._metrics[operation_name] = metrics

            self.logger.log(
                log_level,
                f"Completed operation: {operation_name} "
                f"(Duration: {duration:.2f}s, "
                f"Memory: {start_memory:.1f}MB -> {end_memory:.1f}MB)",
            )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def get_metrics(self, operation_name: str) -> Optional[PerformanceMetrics]:
        """Get metrics for a specific operation."""
        return self._metrics.get(operation_name)

    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get all recorded metrics."""
        return self._metrics.copy()

    def log_summary(self, log_level: int = logging.INFO):
        """Log a summary of all recorded metrics."""
        if not self._metrics:
            return

        total_duration = sum(m.duration for m in self._metrics.values())
        peak_memory = max(m.memory_peak for m in self._metrics.values())

        self.logger.log(log_level, "=== Performance Summary ===")
        self.logger.log(log_level, f"Total operations: {len(self._metrics)}")
        self.logger.log(log_level, f"Total duration: {total_duration:.2f}s")
        self.logger.log(log_level, f"Peak memory usage: {peak_memory:.1f}MB")

        for name, metrics in self._metrics.items():
            self.logger.log(
                log_level,
                f"  {name}: {metrics.duration:.2f}s "
                f"({metrics.memory_start:.1f}MB -> {metrics.memory_end:.1f}MB)",
            )


class LoggingContext:
    """Logging context manager for consistent logging setup."""

    def __init__(self, name: str, config_path: Optional[Path] = None):
        self.name = name
        self.config = get_config(config_path)
        self.logger = self._setup_logger()
        self.performance_logger = PerformanceLogger(self.logger)

    def _setup_logger(self) -> logging.Logger:
        """Setup logger with configuration."""
        logger = logging.getLogger(self.name)

        # Set level
        level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        logger.setLevel(level)

        # Clear existing handlers
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(self.config.logging.format)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

        # Add file handler if configured
        if self.config.logging.file_path:
            file_path = Path(self.config.logging.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)

        # Prevent propagation to root logger
        logger.propagate = False

        return logger

    def get_logger(self) -> logging.Logger:
        """Get the configured logger."""
        return self.logger

    def get_performance_logger(self) -> PerformanceLogger:
        """Get the performance logger."""
        return self.performance_logger


def get_logger(name: str, config_path: Optional[Path] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name
        config_path: Optional path to configuration file

    Returns:
        Configured logger instance
    """
    context = LoggingContext(name, config_path)
    return context.get_logger()


def performance_tracker(operation_name: Optional[str] = None):
    """
    Decorator to track function performance.

    Args:
        operation_name: Optional custom operation name
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get logger from first argument if it's a class instance
            logger = None
            perf_logger = None

            if args and hasattr(args[0], "logger"):
                logger = args[0].logger
            elif args and hasattr(args[0], "_logger"):
                logger = args[0]._logger
            else:
                logger = get_logger(func.__module__)

            if hasattr(logger, "performance_logger"):
                perf_logger = logger.performance_logger
            else:
                perf_logger = PerformanceLogger(logger)

            name = operation_name or f"{func.__module__}.{func.__name__}"

            with perf_logger.track_operation(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class ProgressTracker:
    """Progress tracking utility for long-running operations."""

    def __init__(
        self,
        logger: logging.Logger,
        total_steps: int,
        operation_name: str = "Operation",
    ):
        self.logger = logger
        self.total_steps = total_steps
        self.operation_name = operation_name
        self.current_step = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.log_interval = 5.0  # Log every 5 seconds

    def update(self, step: int, message: str = ""):
        """Update progress."""
        self.current_step = step
        current_time = time.time()

        # Log progress at intervals or on completion
        if (
            current_time - self.last_log_time >= self.log_interval
            or step >= self.total_steps
        ):
            progress_percent = (step / self.total_steps) * 100
            elapsed_time = current_time - self.start_time

            if step > 0:
                estimated_total = elapsed_time * self.total_steps / step
                remaining_time = estimated_total - elapsed_time

                self.logger.info(
                    f"{self.operation_name}: {progress_percent:.1f}% "
                    f"({step}/{self.total_steps}) - "
                    f"Elapsed: {elapsed_time:.1f}s, "
                    f"Remaining: {remaining_time:.1f}s"
                    f"{f' - {message}' if message else ''}"
                )
            else:
                self.logger.info(
                    f"{self.operation_name}: {progress_percent:.1f}% "
                    f"({step}/{self.total_steps})"
                    f"{f' - {message}' if message else ''}"
                )

            self.last_log_time = current_time

    def finish(self, message: str = "Completed"):
        """Mark operation as finished."""
        total_time = time.time() - self.start_time
        self.logger.info(
            f"{self.operation_name}: {message} " f"(Total time: {total_time:.2f}s)"
        )
