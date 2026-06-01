# Record 007 NPU Port

This directory contains the NPU port of the canonical Record 7 `pytorch25` source.

Primary entrypoint:

- `train_gpt2.py`

Supporting scripts:

- `collect_env.py`: writes per-seed runtime metadata.
- `parse_metrics.py`: extracts validation losses from `train.log`.
- `write_run_status.py`: writes per-seed `run_status.json`.
- `update_stats.py`: aggregates counted KTP seed metrics.

The port preserves the Record 7 model, Muon plus AdamW optimizer split, scheduler, binary data loader, global batch size, and validation token count. NPU-specific changes are documented in `../source_audit.md`.
