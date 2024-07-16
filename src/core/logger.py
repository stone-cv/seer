import os
import datetime
import logging

from logging.handlers import TimedRotatingFileHandler

import src.core.config as cfg

# Create a logger
logger = logging.getLogger("my_logger")

# Create a file handler with daily rotation
os.makedirs(cfg.log_dir, exist_ok=True)
# log_file = f"{cfg.log_dir}/log_{datetime.date.today()}.log"
log_file = f"{cfg.log_dir}/log"
file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=30)
file_handler.namer = lambda name: name + ".log"

# Create a stream handler for console logging
console_handler = logging.StreamHandler()

# Create a formatter
formatter = logging.Formatter('%(asctime)s %(name)s:%(levelname)s - %(message)s')

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.setLevel(logging.DEBUG if cfg.DEBUG else logging.INFO)