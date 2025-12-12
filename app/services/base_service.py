# app/services/base_service.py
import logging

class BaseService:
    """
    A base class for all services, providing common functionalities like logging.
    """
    def __init__(self, logger: logging.Logger):
        """
        Initializes the base service.

        Args:
            logger: The logger instance for logging messages.
        """
        self.logger = logger