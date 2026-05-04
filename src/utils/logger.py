import logging

from utils.colored_formatter import ColoredFormatter


class Logger:

    def __init__(self, verbosity: bool = False):
        self.verbosity = verbosity
        self._logger = logging.getLogger("GenderDetector")
        self._logger.propagate = False

        # Clear any existing handlers to avoid duplicates
        self._logger.handlers.clear()

        # Set log level based on verbosity
        level = logging.INFO if verbosity else logging.WARNING
        self._logger.setLevel(level)

        # Create console handler with formatting
        handler = logging.StreamHandler()
        handler.setLevel(level)

        formatter = ColoredFormatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    def info(self, message: str):
        self._logger.info(message)

    def debug(self, message: str):
        self._logger.debug(message)

    def warning(self, message: str):
        self._logger.warning(message)

    def error(self, message: str):
        self._logger.error(message)
