# app/utils/logger_config.py
"""
Centralized logging configuration for SpectroEase.
Replaces scattered print statements with structured logging.
"""

import logging
import sys
from pathlib import Path

# Define log levels for different modules
LOG_LEVELS = {
    'main': logging.INFO,
    'data_flow': logging.INFO,  # Data flow tracking
    'feature_selection': logging.INFO,
    'preprocessing': logging.INFO,
    'modeling': logging.INFO,
    'visualization': logging.WARNING,  # Less verbose for viz
    'debug': logging.DEBUG,  # For development
}

def setup_logger(name='spectroease', level=logging.INFO, log_file=None):
    """
    Setup a logger with consistent formatting.
    
    Args:
        name: Logger name (typically module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(levelname)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # File gets all logs
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(module_name):
    """
    Get or create a logger for a specific module.
    
    Args:
        module_name: Name of the module (e.g., 'feature_selection', 'preprocessing')
        
    Returns:
        Logger instance
    """
    level = LOG_LEVELS.get(module_name, logging.INFO)
    return setup_logger(f'spectroease.{module_name}', level=level)

# Convenience function to reduce verbosity
def log_data_shape(logger, data, name="Data", level=logging.DEBUG):
    """Log shape information for data structures."""
    if data is not None:
        if hasattr(data, 'shape'):
            logger.log(level, f"{name} shape: {data.shape}")
        elif isinstance(data, (list, tuple)):
            logger.log(level, f"{name} length: {len(data)}")
    else:
        logger.log(level, f"{name}: None")

