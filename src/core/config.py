import os
import yaml
from pydantic import PostgresDsn


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


# Application
app_name = config['Application']['app_name']
DEBUG = config['Application']['DEBUG']
log_dir = config['Application']['log_dir']
# results_dir = config['Application']['results_dir']
download_dir = config['Application']['download_dir']
video_path = config['Application']['video_path']
required_fps = config['Application']['required_fps']
deep_archive = config['Application']['deep_archive']
delay = config['Application']['delay']
weights_det = config['Application']['weights_det']
weights_seg = config['Application']['weights_seg']

# API
send_json = config['API_info']['send_json']
send_img = config['API_info']['send_img']
json_url = config['API_info']['json_url']
img_url = config['API_info']['img_url']
json_auth_token = config['API_info']['json_auth_token']

# camera auth
cam_login = config['Camera_auth']['login']
cam_password = config['Camera_auth']['password']
recorder_ip = config['Camera_auth']['recorder_ip']
camera_id = config['Camera_auth']['camera_id']
channel = config['Camera_auth']['channel']

# saw logic
saw_moving_sec = config['Saw_logic']['saw_moving_sec']
saw_moving_threshold = config['Saw_logic']['saw_moving_threshold']

# stone logic
stone_history_threshold = config['Stone_logic']['stone_history_threshold']
stone_change_threshold = config['Stone_logic']['stone_change_threshold']
forklift_history_threshold = config['Stone_logic']['forklift_history_threshold']
forklift_present_threshold = config['Stone_logic']['forklift_present_threshold']
majority_threshold = config['Stone_logic']['majority_threshold']
max_stone_area_list = config['Stone_logic']['max_stone_area_list']

# Database
db_host = config['Database']['db_host']
db_port = config['Database']['db_port']
db_name = config['Database']['db_name']
db_login = config['Database']['db_login']
db_password = config['Database']['db_password']
