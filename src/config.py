import os

import yaml

CONFIG_PATH = "proj_config.yml"
try:
    with open(CONFIG_PATH, 'r') as stream:
        CONFIG = yaml.safe_load(stream)
except FileNotFoundError as e:
    with open(os.path.join('/code', CONFIG_PATH), 'r') as stream:
        CONFIG = yaml.safe_load(stream)
