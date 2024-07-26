import os
import datetime
import logging

from logging.handlers import TimedRotatingFileHandler

import core.config as cfg

logger = logging.getLogger("my_logger")
os.makedirs(cfg.log_dir, exist_ok=True)

# log_file = f"{cfg.log_dir}/log_{datetime.date.today()}.log"
log_file = f"{cfg.log_dir}/log"
file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=30)
file_handler.namer = lambda name: name + ".log"

console_handler = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s:%(levelname)s - %(message)s')

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.setLevel(logging.DEBUG if cfg.DEBUG else logging.INFO)
