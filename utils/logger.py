# utils/logger.py

import logging
import os
import sys

def get_log_path(log_file):
    """Get the absolute path for the log file, compatible with PyInstaller."""
    if getattr(sys, 'frozen', False):
        # Running in a bundle
        base_path = os.path.dirname(sys.executable)
    else:
        # Running in a normal Python environment
        base_path = os.path.abspath(".")
    return os.path.join(base_path, log_file)

def setup_logger(name='SpectroEase1.1', log_file='app.log', level=logging.INFO):
    """Setup logger configuration"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    
    log_path = get_log_path(log_file)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger
