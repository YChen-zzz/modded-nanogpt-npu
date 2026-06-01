import json
import os
import re
import sys


VAL_RE = re.compile(r"step:(?P<step>\d+)/(?:\d+) val_loss:(?P<loss>[0-9.]+)")
TRAIN_RE = re.compile(r"step:(?P<step>\d+)/(?:\d+) train_loss:(?P<loss>[0-9.]+)")


def main():
    log_path = sys.argv[1]
    metrics_path = sys.argv[2]
    exit_code = int(sys.argv[3]) if len(sys.argv) > 3 else None
    val_points = []
    train_points = []
    if os.path.exists(log_path):
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                val_match = VAL_RE.search(line)
                if val_match:
                    val_points.append({
                        "step": int(val_match.group("step")),
                        "val_loss": float(val_match.group("loss")),
                    })
                train_match = TRAIN_RE.search(line)
                if train_match:
                    train_points.append({
                        "step": int(train_match.group("step")),
                        "train_loss": float(train_match.group("loss")),
                    })
    final_val = val_points[-1]["val_loss"] if val_points else None
    metrics = {
        "schema_version": 1,
        "exit_code": exit_code,
        "val_loss": final_val,
        "final_val_loss": final_val,
        "validation_points": val_points,
        "final_train_loss": train_points[-1]["train_loss"] if train_points else None,
        "train_points_count": len(train_points),
        "log_path": log_path,
    }
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
