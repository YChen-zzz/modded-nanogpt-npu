import json
import os
import sys


env_path = sys.argv[1]
job_id = sys.argv[2]
if os.path.exists(env_path):
    with open(env_path) as f:
        payload = json.load(f)
else:
    payload = {"schema_version": 1}
payload["job_id"] = job_id
with open(env_path, "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
