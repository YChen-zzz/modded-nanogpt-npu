import json
import os
import sys


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    run_dir = sys.argv[1]
    seed = int(sys.argv[2])
    exit_code = int(sys.argv[3])
    metrics = load_json(os.path.join(run_dir, "metrics.json")) or {}
    has_job_id = os.path.exists(os.path.join(run_dir, "job_id.txt"))
    status = "COMPLETED" if exit_code == 0 and metrics.get("val_loss") is not None else "FAILED"
    payload = {
        "schema_version": 1,
        "seed": seed,
        "status": status,
        "exit_code": exit_code,
        "job_id_present": has_job_id,
        "val_loss": metrics.get("val_loss"),
        "metrics_path": "metrics.json",
    }
    with open(os.path.join(run_dir, "run_status.json"), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
