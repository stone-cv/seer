import os
import datetime
import logging

from logging.handlers import TimedRotatingFileHandler

import core.config as cfg

# Create a logger
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG if cfg.DEBUG else logging.INFO)

# Create a file handler with daily rotation
os.makedirs(cfg.log_dir, exist_ok=True)
log_file = f"{cfg.log_dir}/log_{datetime.date.today()}.log"
file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7)
# file_handler.setLevel(logging.DEBUG if config.DEBUG else logging.INFO)

# Create a stream handler for console logging
console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG if config.DEBUG else logging.INFO)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Add the file handler to the logger
logger.addHandler(file_handler)
