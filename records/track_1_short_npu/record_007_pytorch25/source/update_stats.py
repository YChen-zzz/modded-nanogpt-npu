import json
import math
import os
import re
import statistics
import sys


THRESHOLD = 3.28
MAX_SEEDS = 10
VAL_RE = re.compile(r"step:(?P<step>\d+)/(?:\d+) val_loss:(?P<loss>[0-9.]+)")


def ttest_less(losses, threshold):
    if len(losses) < 2:
        return None
    mean = statistics.fmean(losses)
    std = statistics.stdev(losses)
    if std == 0:
        return 0.0 if mean < threshold else 1.0
    try:
        from scipy import stats

        return float(stats.ttest_1samp(losses, threshold, alternative="less").pvalue)
    except Exception:
        return None


def load_json(path):
    with open(path) as f:
        return json.load(f)


def latest_partial_val_loss(run_dir):
    log_path = os.path.join(run_dir, "train.log")
    if not os.path.exists(log_path):
        return None
    latest = None
    with open(log_path, errors="replace") as f:
        for line in f:
            match = VAL_RE.search(line)
            if match:
                latest = {
                    "step": int(match.group("step")),
                    "val_loss": float(match.group("loss")),
                }
    return latest


def main():
    output_dir = sys.argv[1]
    decision = sys.argv[2] if len(sys.argv) > 2 else "RUNNING"
    runs_dir = os.path.join(output_dir, "runs")
    seeds = []
    losses = []
    job_ids = []
    if os.path.isdir(runs_dir):
        for name in sorted(os.listdir(runs_dir)):
            if not name.startswith("seed_"):
                continue
            run_dir = os.path.join(runs_dir, name)
            metrics_path = os.path.join(run_dir, "metrics.json")
            run_status_path = os.path.join(run_dir, "run_status.json")
            job_id_path = os.path.join(run_dir, "job_id.txt")
            ktp_get_path = os.path.join(run_dir, "ktp_get.json")
            ktp_logs_path = os.path.join(run_dir, "ktp_logs_tail.txt")
            metrics = load_json(metrics_path) if os.path.exists(metrics_path) else {}
            run_status = load_json(run_status_path) if os.path.exists(run_status_path) else {}
            ktp_get = load_json(ktp_get_path) if os.path.exists(ktp_get_path) else {}
            job_id = open(job_id_path).read().strip() if os.path.exists(job_id_path) else None
            val_loss = metrics.get("val_loss")
            has_ktp_evidence = bool(job_id) and os.path.exists(ktp_get_path) and os.path.exists(ktp_logs_path)
            seed_num = int(name.split("_", 1)[1])
            status = run_status.get("status")
            if not status:
                ktp_status = ktp_get.get("status")
                if ktp_status in ("Running", "Pending"):
                    status = ktp_status.upper()
                elif ktp_status in ("Completed", "Succeeded"):
                    status = "COMPLETED_NEEDS_METRICS"
                elif ktp_status:
                    status = ktp_status.upper()
                else:
                    status = "PENDING"
            seeds.append({
                "seed": seed_num,
                "status": status,
                "job_id": job_id,
                "val_loss": val_loss,
                "latest_partial_val_loss": latest_partial_val_loss(run_dir),
                "artifact_dir": f"runs/{name}",
                "has_ktp_evidence": has_ktp_evidence,
            })
            if has_ktp_evidence and val_loss is not None and run_status.get("status") == "COMPLETED":
                losses.append(float(val_loss))
                job_ids.append(job_id)
    n = len(losses)
    mean = statistics.fmean(losses) if losses else None
    std = statistics.stdev(losses) if len(losses) >= 2 else None
    p_value = ttest_less(losses, THRESHOLD)
    seed_budget_exhausted = n >= MAX_SEEDS and not (
        mean is not None and mean <= THRESHOLD and p_value is not None and p_value < 0.01
    )
    stats_payload = {
        "schema_version": 1,
        "record_id": 7,
        "threshold": THRESHOLD,
        "alternative": "less",
        "code_config_id": "default",
        "max_counted_seeds_per_code_config": MAX_SEEDS,
        "seed_budget_exhausted": seed_budget_exhausted,
        "losses": losses,
        "n": n,
        "mean_val_loss": mean,
        "std_val_loss": std,
        "p_value": p_value,
        "passes_loss_threshold": bool(mean is not None and mean <= THRESHOLD),
        "passes_significance": bool(p_value is not None and p_value < 0.01),
        "decision": decision,
        "seeds": seeds,
        "job_ids": job_ids,
    }
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats_payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
