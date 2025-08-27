# utils/logger.py

import logging
import os

def setup_logger(name='SpectroEase1.1', log_file='app.log', level=logging.INFO):
    """Setup logger configuration"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

  
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger
