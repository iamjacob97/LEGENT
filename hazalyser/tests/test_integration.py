import json
import os
from dotenv import dotenv_values
from hazalyser.helpers import ENV_VARS_PATH, get_env_key

print(ENV_VARS_PATH)
vars = dotenv_values(ENV_VARS_PATH)
print(vars.keys())

for key in vars.keys():
    raw = os.environ.get(key)
    print(type(json.loads(raw).get("vision_support")))