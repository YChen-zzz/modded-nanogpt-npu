import json
import re
import sys


text = open(sys.argv[1], errors="replace").read()
try:
    payload = json.loads(text)
    for key in ("id", "job_id", "jobId", "uid"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if value is not None and re.fullmatch(r"\d+", str(value)):
            print(value)
            raise SystemExit(0)
except Exception:
    pass

patterns = [
    r'"(?:id|job_id|jobId)"\s*:\s*"?(\d+)"?',
    r'\bjob(?:\s+id|_id|Id)?\b[^0-9]{0,20}(\d{3,})',
    r'\bid\b[^0-9]{0,20}(\d{3,})',
]
for pattern in patterns:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        print(match.group(1))
        raise SystemExit(0)

raise SystemExit(1)
