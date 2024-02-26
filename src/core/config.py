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
required_fps = config['Application']['required_fps']
weights_path = config['Application']['weights_path']
deep_archive = config['Application']['deep_archive']
delay = config['Application']['delay']

# camera auth
cam_login = config['Camera_auth']['login']
cam_password = config['Camera_auth']['password']
recorder_ip = config['Camera_auth']['recorder_ip']
channel = config['Camera_auth']['channel']

# saw logic
saw_moving_sec = config['Saw_logic']['saw_moving_sec']
saw_moving_threshold = config['Saw_logic']['saw_moving_threshold']

# stone logic
stone_check_sec = config['Stone_logic']['stone_check_sec']

# Database
db_host = config['Database']['db_host']
db_port = config['Database']['db_port']
db_name = config['Database']['db_name']
db_login = config['Database']['db_login']
db_password = config['Database']['db_password']
