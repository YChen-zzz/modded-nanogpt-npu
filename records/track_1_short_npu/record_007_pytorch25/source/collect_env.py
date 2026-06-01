import json
import os
import platform
import subprocess
import sys


def command_output(argv):
    try:
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout[-8000:],
            "stderr": result.stderr[-8000:],
        }
    except Exception as exc:
        return {"error": repr(exc)}


def main():
    env_path = sys.argv[1]
    command = sys.argv[2] if len(sys.argv) > 2 else ""
    info = {
        "schema_version": 1,
        "image": "docker.cnb.cool/nilpotenter/docker/codeserver-mindspeed",
        "image_tag": "v1.0.5",
        "queue": "user-zhaokunxiang-compute",
        "framework": "PyTorch",
        "job_type": "vcjob",
        "npu_request": 16,
        "NPROC": os.environ.get("NPROC"),
        "world_size": os.environ.get("WORLD_SIZE"),
        "job_id": os.environ.get("KTP_JOB_ID") or os.environ.get("JOB_ID"),
        "command": command,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "env": {
            key: os.environ.get(key)
            for key in [
                "ASCEND_VISIBLE_DEVICES",
                "ASCEND_RT_VISIBLE_DEVICES",
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "HCCL_CONNECT_TIMEOUT",
                "PYTORCH_NPU_ALLOC_CONF",
                "OUTPUT_DIR",
                "RUN_DIR",
                "NANOGPT_SEED",
                "DEVICE_BATCH_SIZE",
                "NUM_ITERATIONS",
                "VAL_LOSS_EVERY",
                "SAVE_EVERY",
                "ENABLE_TORCH_COMPILE",
            ]
        },
        "npu_smi": command_output(["npu-smi", "info"]),
    }
    try:
        import torch

        info["torch"] = getattr(torch, "__version__", None)
        info["npu_available"] = bool(torch.npu.is_available())
        info["npu_count"] = int(torch.npu.device_count()) if torch.npu.is_available() else 0
    except Exception as exc:
        info["torch_error"] = repr(exc)
    try:
        import torch_npu

        info["torch_npu"] = getattr(torch_npu, "__version__", None)
    except Exception as exc:
        info["torch_npu_error"] = repr(exc)

    os.makedirs(os.path.dirname(env_path), exist_ok=True)
    with open(env_path, "w") as f:
        json.dump(info, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
