import argparse
import copy
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product

import yaml

try:
    import submitit
except Exception as e:  # pragma: no cover
    submitit = None
    _submitit_import_error = e


@dataclass(frozen=True)
class GridCell:
    target: str
    kl_coefficient: float
    rank_coefficient: float


def _require_submitit():
    if submitit is None:  # pragma: no cover
        raise ImportError(
            "submitit is not installed in this environment. Install it (e.g. `pip install submitit`) "
            f"or run on a cluster image that includes it. Original error: {_submitit_import_error}"
        )


def _abs_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _script_dirs():
    """
    This file lives in: <REPO_ROOT>/multi_objective/submit_hit_task_grid_submitit.py
    """
    script_path = _abs_path(__file__)
    multi_objective_dir = os.path.dirname(script_path)
    repo_root = os.path.dirname(multi_objective_dir)
    return repo_root, multi_objective_dir


def parse_hparam_grid(hparam_config_path: str):
    """
    Expects a yaml like multi_objective/genetic_gfn/hparams_tune.yaml:
      parameters:
        kl_coefficient:
          values: [...]
        rank_coefficient:
          values: [...]
    """
    with open(hparam_config_path, "r") as f:
        hp = yaml.safe_load(f)

    params = (hp or {}).get("parameters", {}) or {}
    kl_vals = ((params.get("kl_coefficient") or {}).get("values")) or []
    rank_vals = ((params.get("rank_coefficient") or {}).get("values")) or []

    if not kl_vals or not rank_vals:
        raise ValueError(
            f"Invalid hparam config at {hparam_config_path}. "
            "Expected parameters.kl_coefficient.values and parameters.rank_coefficient.values."
        )

    # ensure float
    kl_vals = [float(x) for x in kl_vals]
    rank_vals = [float(x) for x in rank_vals]
    return kl_vals, rank_vals


def make_grid_cells(targets, kl_vals, rank_vals):
    cells = []
    for target, kl, rank in product(targets, kl_vals, rank_vals):
        cells.append(GridCell(target=target, kl_coefficient=float(kl), rank_coefficient=float(rank)))
    return cells


def write_config(base_config_path: str, out_config_path: str, kl: float, rank: float):
    with open(base_config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg = copy.deepcopy(cfg)
    cfg["kl_coefficient"] = float(kl)
    cfg["rank_coefficient"] = float(rank)

    os.makedirs(os.path.dirname(out_config_path), exist_ok=True)
    with open(out_config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_one(cell: GridCell, *, repo_root: str, out_dir: str, max_oracle_calls: int, n_runs: int, freq_log: int):
    """
    Runs ONE grid cell (one target + one (kl, rank)) as a single job.

    - Uses multi_objective/run.py genetic_gfn
    - Uses --task production (optimizer seeds are hard-coded to 0..9 in this repo)
    - Writes outputs under OUT_DIR/genetic_gfn/results/...
    """
    _, multi_objective_dir = _script_dirs()

    # Ensure relative paths in the repo work (targets, docking tmp, etc.)
    original_cwd = os.getcwd()
    os.chdir(multi_objective_dir)

    try:
        results_root = os.path.join(out_dir, "genetic_gfn", "results")
        cell_out_dir = os.path.join(
            results_root,
            "grid",
            cell.target,
            f"kl_{cell.kl_coefficient}",
            f"rank_{cell.rank_coefficient}",
        )
        os.makedirs(cell_out_dir, exist_ok=True)

        base_config = os.path.join(multi_objective_dir, "genetic_gfn", "hparams_default.yaml")
        cfg_out = os.path.join(cell_out_dir, f"hparams_kl_{cell.kl_coefficient}_rank_{cell.rank_coefficient}.yaml")
        write_config(base_config, cfg_out, cell.kl_coefficient, cell.rank_coefficient)

        run_name = f"{cell.target}_hit_task_kl{cell.kl_coefficient}_rank{cell.rank_coefficient}"
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
            cell_out_dir,
            "--config_default",
            cfg_out,
        ]

        print("CWD:", os.getcwd())
        print("CMD:", " ".join(cmd))
        subprocess.run(cmd, check=False)
    finally:
        try:
            os.chdir(original_cwd)
        except Exception:
            pass


def main():
    _require_submitit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["parp1", "fa7", "5ht1b"])
    parser.add_argument("--hparam_config", type=str, default="genetic_gfn/hparams_tune.yaml")
    parser.add_argument("--max_oracle_calls", type=int, default=3000)
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--freq_log", type=int, default=100)

    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--partition", type=str, default="batch")
    parser.add_argument("--cpus_per_task", type=int, default=12)
    parser.add_argument("--mem_gb", type=int, default=32)
    parser.add_argument("--timeout_min", type=int, default=48 * 60)
    parser.add_argument("--slurm_array_parallelism", type=int, default=9)
    parser.add_argument("--job_name", type=str, default="genetic_gfn_hit_grid")
    parser.add_argument("--direct", action="store_true", default=False)

    args = parser.parse_args()

    out_dir = os.environ.get("OUT_DIR")
    if not out_dir:
        raise ValueError("OUT_DIR environment variable is required (root for results + submitit logs).")
    out_dir = _abs_path(out_dir)

    repo_root, multi_objective_dir = _script_dirs()

    # Make hparam_config absolute (relative to multi_objective/)
    hparam_config_path = args.hparam_config
    if not os.path.isabs(hparam_config_path):
        hparam_config_path = os.path.join(multi_objective_dir, hparam_config_path)

    kl_vals, rank_vals = parse_hparam_grid(hparam_config_path)
    cells = make_grid_cells(args.targets, kl_vals, rank_vals)

    # Submitit folder (logs) rooted at OUT_DIR
    ts = time.strftime("%Y%m%d_%H%M%S")
    submitit_root = os.path.join(out_dir, "genetic_gfn", "slurm_jobs", "submitit", "multi_objective_hit_grid", ts)
    os.makedirs(submitit_root, exist_ok=True)

    if args.direct:
        executor = submitit.LocalExecutor(folder=os.path.join(submitit_root, "%j"))
        executor.update_parameters(
            timeout_min=args.timeout_min,
            gpus_per_node=args.n_gpus,
            nodes=1,
            mem_gb=args.mem_gb,
            cpus_per_task=args.cpus_per_task,
        )
    else:
        executor = submitit.AutoExecutor(folder=os.path.join(submitit_root, "%j"))
        executor.update_parameters(
            slurm_job_name=args.job_name,
            timeout_min=args.timeout_min,
            slurm_array_parallelism=args.slurm_array_parallelism,
            gpus_per_node=args.n_gpus,
            nodes=1,
            mem_gb=args.mem_gb,
            cpus_per_task=args.cpus_per_task,
            slurm_additional_parameters={"partition": args.partition},
        )

    print("REPO_ROOT:", repo_root)
    print("MULTI_OBJECTIVE_DIR:", multi_objective_dir)
    print("OUT_DIR:", out_dir)
    print("SUBMITIT_LOG_ROOT:", submitit_root)
    print("NUM_JOBS:", len(cells))

    jobs = []
    submitted = []
    with executor.batch():
        for cell in cells:
            job = executor.submit(
                run_one,
                cell=cell,
                repo_root=repo_root,
                out_dir=out_dir,
                max_oracle_calls=args.max_oracle_calls,
                n_runs=args.n_runs,
                freq_log=args.freq_log,
            )
            # NOTE: submitit forbids accessing job.job_id inside batch() context.
            submitted.append((job, cell))

    # After exiting batch(), job ids are available.
    for job, cell in submitted:
        print("submitted:", job.job_id, cell)
        jobs.append(job)

    print(f"Submitted {len(jobs)} jobs.")


if __name__ == "__main__":
    main()

