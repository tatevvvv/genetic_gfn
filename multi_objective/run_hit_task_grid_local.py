#!/usr/bin/env python3
"""
Local multi-GPU "job imitation" runner for multi_objective genetic_gfn hit-task grid search.

Use when Slurm is unavailable but you have a multi-GPU node (e.g., 8x H100).

It enumerates the 3x3x3 grid:
  targets (default): parp1, fa7, 5ht1b
  kl_coefficient values and rank_coefficient values from genetic_gfn/hparams_tune.yaml

For each grid cell it runs:
  python run.py genetic_gfn --objectives qed,sa,<target> --alpha_vector 1,1,1
    --max_oracle_calls 3000 --task production --n_runs 10 --freq_log 100 --wandb disabled

Concurrency:
  Runs up to N processes in parallel and pins each process to a GPU via CUDA_VISIBLE_DEVICES.

Output:
  Requires OUT_DIR env var. Writes:
    - results/CSVs:  $OUT_DIR/genetic_gfn/results/grid/<target>/kl_<..>/rank_<..>/
    - logs:          $OUT_DIR/genetic_gfn/local_jobs/multi_objective_hit_grid/<run_name>.gpu<G>.log
  Also uses a per-run DOCKING_TMP_DIR under the run’s output folder to avoid collisions.
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import yaml


@dataclass(frozen=True)
class Cell:
    target: str
    kl: float
    rank: float


def abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def dirs() -> Tuple[str, str]:
    # <REPO_ROOT>/multi_objective/run_hit_task_grid_local.py
    script_path = abspath(__file__)
    multi_obj_dir = os.path.dirname(script_path)
    repo_root = os.path.dirname(multi_obj_dir)
    return repo_root, multi_obj_dir


def parse_hparams(hparam_yaml: str) -> Tuple[List[float], List[float]]:
    with open(hparam_yaml, "r") as f:
        hp = yaml.safe_load(f) or {}
    params = (hp.get("parameters") or {}) if isinstance(hp, dict) else {}
    kl_vals = (((params.get("kl_coefficient") or {}).get("values")) or [])
    rank_vals = (((params.get("rank_coefficient") or {}).get("values")) or [])
    if not kl_vals or not rank_vals:
        raise ValueError(
            f"Invalid hparam config: {hparam_yaml}. "
            "Expected parameters.kl_coefficient.values and parameters.rank_coefficient.values."
        )
    return [float(x) for x in kl_vals], [float(x) for x in rank_vals]


def write_config(base_cfg: str, out_cfg: str, *, kl: float, rank: float) -> None:
    with open(base_cfg, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["kl_coefficient"] = float(kl)
    cfg["rank_coefficient"] = float(rank)
    os.makedirs(os.path.dirname(out_cfg), exist_ok=True)
    with open(out_cfg, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def build_cells(targets: List[str], kl_vals: List[float], rank_vals: List[float]) -> List[Cell]:
    return [Cell(t, float(kl), float(rank)) for t, kl, rank in product(targets, kl_vals, rank_vals)]


def launch(
    *,
    cell: Cell,
    gpu_id: int,
    multi_obj_dir: str,
    out_dir: str,
    max_oracle_calls: int,
    n_runs: int,
    freq_log: int,
) -> subprocess.Popen:
    results_root = os.path.join(out_dir, "genetic_gfn", "results")
    logs_root = os.path.join(out_dir, "genetic_gfn", "local_jobs", "multi_objective_hit_grid")
    os.makedirs(results_root, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)

    run_name = f"{cell.target}_hit_task_kl{cell.kl}_rank{cell.rank}"
    run_out = os.path.join(results_root, "grid", cell.target, f"kl_{cell.kl}", f"rank_{cell.rank}")
    os.makedirs(run_out, exist_ok=True)

    # Per-run docking tmp dir to avoid collisions across parallel processes
    docking_tmp_dir = os.path.join(run_out, "docking_tmp")
    os.makedirs(docking_tmp_dir, exist_ok=True)

    base_cfg = os.path.join(multi_obj_dir, "genetic_gfn", "hparams_default.yaml")
    cfg_out = os.path.join(run_out, f"hparams_kl_{cell.kl}_rank_{cell.rank}.yaml")
    write_config(base_cfg, cfg_out, kl=cell.kl, rank=cell.rank)

    cmd = [
        sys.executable,
        "run.py",
        "genetic_gfn",
        "--objectives",
        f"qed,sa,{cell.target}",
        "--alpha_vector",
        "1,1,1",
        "--max_oracle_calls",
        str(max_oracle_calls),
        "--task",
        "production",
        "--n_runs",
        str(n_runs),
        "--freq_log",
        str(freq_log),
        "--wandb",
        "disabled",
        "--run_name",
        run_name,
        "--output_dir",
        run_out,
        "--config_default",
        cfg_out,
    ]

    log_path = os.path.join(logs_root, f"{run_name}.gpu{gpu_id}.log")
    log_f = open(log_path, "w")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["DOCKING_TMP_DIR"] = docking_tmp_dir

    log_f.write(f"CWD: {multi_obj_dir}\n")
    log_f.write(f"CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}\n")
    log_f.write(f"DOCKING_TMP_DIR: {env['DOCKING_TMP_DIR']}\n")
    log_f.write("CMD: " + " ".join(cmd) + "\n\n")
    log_f.flush()

    p = subprocess.Popen(cmd, cwd=multi_obj_dir, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    p._log_f = log_f  # type: ignore[attr-defined]
    p._log_path = log_path  # type: ignore[attr-defined]
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["parp1", "fa7", "5ht1b"])
    ap.add_argument("--hparam_config", default="genetic_gfn/hparams_tune.yaml")
    ap.add_argument("--max_oracle_calls", type=int, default=3000)
    ap.add_argument("--n_runs", type=int, default=10)
    ap.add_argument("--freq_log", type=int, default=100)

    ap.add_argument("--gpu_ids", nargs="*", type=int, default=None, help="Explicit GPU ids (e.g., 0 1 2 3 4 5 6 7)")
    ap.add_argument("--num_gpus", type=int, default=8, help="Used only if --gpu_ids not provided.")
    ap.add_argument("--max_parallel", type=int, default=8)
    ap.add_argument("--poll_sec", type=float, default=2.0)
    args = ap.parse_args()

    out_dir = os.environ.get("OUT_DIR")
    if not out_dir:
        raise ValueError("OUT_DIR env var is required (root for results + logs).")
    out_dir = abspath(out_dir)

    _, multi_obj_dir = dirs()

    hparam_path = args.hparam_config
    if not os.path.isabs(hparam_path):
        hparam_path = os.path.join(multi_obj_dir, hparam_path)
    kl_vals, rank_vals = parse_hparams(hparam_path)
    cells = build_cells(args.targets, kl_vals, rank_vals)

    gpu_ids = args.gpu_ids if args.gpu_ids else list(range(args.num_gpus))
    max_parallel = min(args.max_parallel, len(gpu_ids))

    print("OUT_DIR:", out_dir)
    print("MULTI_OBJECTIVE_DIR:", multi_obj_dir)
    print("NUM_CELLS:", len(cells))
    print("GPU_IDS:", gpu_ids)
    print("MAX_PARALLEL:", max_parallel)

    queue: List[Cell] = list(cells)
    running: List[Tuple[subprocess.Popen, int, Cell]] = []

    def poll_finished():
        nonlocal running
        still = []
        for p, gid, cell in running:
            rc = p.poll()
            if rc is None:
                still.append((p, gid, cell))
                continue
            try:
                p._log_f.flush()  # type: ignore[attr-defined]
                p._log_f.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"[done] {status} gpu={gid} cell={cell} log={getattr(p, '_log_path', '')}")
        running = still

    while queue or running:
        poll_finished()

        while queue and len(running) < max_parallel:
            used = {gid for _, gid, _ in running}
            free = [g for g in gpu_ids if g not in used]
            if not free:
                break
            gid = free[0]
            cell = queue.pop(0)
            p = launch(
                cell=cell,
                gpu_id=gid,
                multi_obj_dir=multi_obj_dir,
                out_dir=out_dir,
                max_oracle_calls=args.max_oracle_calls,
                n_runs=args.n_runs,
                freq_log=args.freq_log,
            )
            print(f"[start] gpu={gid} pid={p.pid} cell={cell}")
            running.append((p, gid, cell))

        time.sleep(args.poll_sec)

    print("All runs finished.")


if __name__ == "__main__":
    main()

