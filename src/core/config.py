import os
import yaml
from pydantic import PostgresDsn

# from logger import logger  -- circular import


# load config from env
if os.getenv('CONFIG', None) is None:
    print("Does not exists config file. Use CONFIG env.")
    exit()
else:
    config_path = os.getenv("CONFIG")
    print(os.getenv("CONFIG"))


with open(config_path,"r") as file_object:
    config=yaml.load(file_object, Loader=yaml.SafeLoader)

# config_dict[section_name].update({field_name:field_value})

app_name = config['Application']['app_name']
DEBUG = config['Application']['DEBUG']
log_dir = config['Application']['log_dir']
results_dir = config['Application']['results_dir']
video_path = config['Application']['video_path']
camera_1_roi = config['Application']['camera_1_roi']

# saw logic
saw_moving_sec = config['Saw_logic']['saw_moving_sec']
saw_moving_threshold = config['Saw_logic']['saw_moving_threshold']

# stone logic
stone_check_sec = config['Stone_logic']['stone_check_sec']

# Database
# bot_db_host = config.get('Bot_Database', 'host')
# bot_db_port = config.get('Bot_Database', 'port')
# bot_db_name = config.get('Bot_Database', 'db_name')
# bot_db_login = config.get('Bot_Database', 'login')
# bot_db_password = config.get('Bot_Database', 'password')

# bot_db_URI = PostgresDsn.build(
#     scheme='postgresql+asyncpg',
#     user=bot_db_login,
#     password=bot_db_password,
#     host=bot_db_host,
#     port=bot_db_port,
#     path=f"/{bot_db_name}"
# )
