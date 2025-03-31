import yaml
import os
from pathlib import Path

IS_DEBUG = True

# Get the directory containing parse_config.py
BASE_DIR = Path(__file__).parent

# Load the YAML file from the same directory
config_path = BASE_DIR / 'config.yaml'

# Load the YAML file
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Function to replace placeholders in the config
def replace_placeholders(config):
    for key, value in config.items():
        if isinstance(value, str):
            # Replace placeholders with actual values from the config
            config[key] = value.replace('${OLD_VERSION}', config.get('OLD_VERSION', ''))
            config[key] = config[key].replace('${VERSION}', config.get('VERSION', ''))
    return config

# Replace placeholders
config = replace_placeholders(config)

# Convert THRESHOLD to float if it's a string
if 'THRESHOLD' in config and isinstance(config['THRESHOLD'], str):
    config['THRESHOLD'] = float(config['THRESHOLD'])

if IS_DEBUG:
    print("\nConfig:")
    print(config)
    print("\n")