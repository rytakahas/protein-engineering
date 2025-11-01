#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import yaml
import optuna

EPOCH_LINE = re.compile(
    r"^\[epoch\s+(\d+)\]\s+train=([0-9.]+)\s+val=([0-9.]+)\s+ROC=([0-9.]+|na)\s+PR=([0-9.]+|na)\s+F1=([0-9.]+|na)"
)

def parse_metrics(stdout: str):
    """Return (score, kind). Prefer best F1, then best PR, else negative val loss.
       If no epoch lines are found, return (float('-inf'), 'none')."""
    best_f1 = None
    best_pr = None
    best_neg_val = None

    for line in stdout.splitlines():
        m = EPOCH_LINE.match(line.strip())
        if not m:
            continue
        _, tr, va, roc, pr, f1 = m.groups()
        try:
            val_loss = float(va)
        except Exception:
            val_loss = float("inf")
        try:
            pr_val = float(pr) if pr != "na" else float("nan")
        except Exception:
            pr_val = float("nan")
        try:
            f1_val = float(f1) if f1 != "na" else float("nan")
        except Exception:
            f1_val = float("nan")

        # track bests
        if not (f1_val != f1_val):  # not NaN
            best_f1 = f1_val if (best_f1 is None or f1_val > best_f1) else best_f1
        if not (pr_val != pr_val):  # not NaN
            best_pr = pr_val if (best_pr is None or pr_val > best_pr) else best_pr
        neg_val = -val_loss
        best_neg_val = neg_val if (best_neg_val is None or neg_val > best_neg_val) else best_neg_val

    if best_f1 is not None:
        return best_f1, "f1"
    if best_pr is not None:
        return best_pr, "pr"
    if best_neg_val is not None:
        return best_neg_val, "neg_val"
    return float("-inf"), "none"


def run_one_trial(trial_cfg: dict, script_path: Path, extra_args, log_dir: Path, trial_number: int):
    # write temp YAML for this trial
    with tempfile.TemporaryDirectory() as td:
        tmp_cfg_path = Path(td) / "trial.yaml"
        with open(tmp_cfg_path, "w") as f:
            yaml.safe_dump(trial_cfg, f, sort_keys=False)

        cmd = [sys.executable, str(script_path), "--config", str(tmp_cfg_path)]
        cmd += extra_args

        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        stdout = proc.stdout

        # persist logs so you can inspect failures
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"trial_{trial_number:03d}_{stamp}.log"
        with open(log_path, "w") as lf:
            lf.write(stdout)

        if proc.returncode != 0:
            # failed run → no metrics
            return float("-inf"), "fail", stdout, str(log_path)

        metric, kind = parse_metrics(stdout)
        return metric, kind, stdout, str(log_path)


def make_trial_config(base_cfg: dict, hidden_dim: int, lr: float,
                      maybe_dropout: float | None, maybe_thresh: float | None):
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # deep copy via yaml
    cfg["model"]["hidden_dim"] = int(hidden_dim)
    cfg["training"]["lr"] = float(lr)
    if maybe_dropout is not None:
        cfg["model"]["dropout_p"] = float(maybe_dropout)  # train.py reads this
    if maybe_thresh is not None:
        cfg.setdefault("inference", {})["threshold"] = float(maybe_thresh)
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Optuna sweep for rescontact.")
    ap.add_argument("--config", default="configs/rescontact.yaml", help="Base YAML config.")
    ap.add_argument("--script", default="scripts/train.py", help="Training entry script.")
    ap.add_argument("--trials", type=int, default=20, help="Number of trials.")
    ap.add_argument("--timeout", type=int, default=None, help="Global timeout (seconds).")
    ap.add_argument("--study", default=None, help="Optuna storage URL (e.g., sqlite:///rescontact.db).")
    ap.add_argument("--study-name", default="rescontact-sweep", help="Study name.")
    ap.add_argument("--seed", type=int, default=42)

    # laptop-speed knobs
    ap.add_argument("--epochs", type=int, default=6, help="Override epochs per trial.")
    ap.add_argument("--max-train-batches", type=int, default=60, help="Cap batches/epoch.")
    ap.add_argument("--batch-size", type=int, default=None)

    # your convenience knobs → forwarded to train.py
    ap.add_argument("--min-train-examples", type=int, default=None, help="Alias of train.py --max_train_examples.")
    ap.add_argument("--val-split", type=float, default=None, help="Alias of train.py --train_val_split.")

    # search spaces
    ap.add_argument("--tune-hidden", action="store_true", default=True)
    ap.add_argument("--tune-lr", action="store_true", default=True)
    ap.add_argument("--tune-dropout", action="store_true", default=False)
    ap.add_argument("--tune-threshold", action="store_true", default=False)

    ap.add_argument("--space-hidden", nargs="+", type=int, default=[64, 96, 128, 256])
    ap.add_argument("--space-lr", nargs="+", type=float, default=[1e-3, 1.5e-3, 5e-4])
    ap.add_argument("--space-dropout", nargs="+", type=float, default=[0.0, 0.1, 0.2])
    ap.add_argument("--thresh-min", type=float, default=0.3)
    ap.add_argument("--thresh-max", type=float, default=0.7)
    ap.add_argument("--thresh-step", type=float, default=0.05)

    ap.add_argument("--save-best-config", default=None,
                    help="Path to save tuned YAML (default: <base>.tuned.yaml).")
    ap.add_argument("--logs-dir", default="sweeps/logs", help="Where to write per-trial logs.")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(open(args.config))
    base_cfg["project"]["seed"] = int(args.seed)
    if args.epochs is not None:
        base_cfg["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        base_cfg["training"]["batch_size"] = int(args.batch_size)

    # Prepare Optuna study
    if args.study:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.study,
            direction="maximize",
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction="maximize")

    # Extra CLI args for train.py (forward your aliases properly)
    extra_args = []
    if args.epochs is not None:
        extra_args += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        extra_args += ["--batch_size", str(args.batch_size)]
    if args.max_train_batches is not None:
        extra_args += ["--max_train_batches", str(args.max_train_batches), "--debug"]
    if args.min_train_examples is not None:
        extra_args += ["--max_train_examples", str(args.min_train_examples)]
    if args.val_split is not None:
        extra_args += ["--train_val_split", str(args.val_split)]

    script_path = Path(args.script)
    log_dir = Path(args.logs_dir)

    def objective(trial: optuna.Trial):
        # sample hyperparams
        hidden_dim = trial.suggest_categorical("hidden_dim", args.space_hidden) if args.tune_hidden else base_cfg["model"]["hidden_dim"]
        lr = trial.suggest_categorical("lr", args.space_lr) if args.tune_lr else base_cfg["training"]["lr"]
        dropout = trial.suggest_categorical("dropout_p", args.space_dropout) if args.tune_dropout else None

        thresh = None
        if args.tune_threshold:
            grid = []
            v = args.thresh_min
            while v <= args.thresh_max + 1e-9:
                grid.append(round(v, 3))
                v += args.thresh_step
            thresh = trial.suggest_categorical("threshold", grid)

        trial_cfg = make_trial_config(base_cfg, hidden_dim, lr, dropout, thresh)
        metric, kind, stdout, log_path = run_one_trial(trial_cfg, script_path, extra_args, log_dir, trial.number)
        trial.set_user_attr("metric_kind", kind)
        trial.set_user_attr("log_path", log_path)
        # helpful on dashboard
        tail = "\n".join(stdout.splitlines()[-8:])
        trial.set_user_attr("stdout_tail", tail)
        return metric

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    print("\n=== Optuna Results ===")
    print(f"Best value = {study.best_value:.4f} (direction=maximize)")
    print("Best params:", study.best_trial.params)
    print("Metric kind:", study.best_trial.user_attrs.get("metric_kind", "unknown"))
    print("Best trial log:", study.best_trial.user_attrs.get("log_path", "n/a"))

    # Save tuned YAML
    best_cfg = make_trial_config(
        base_cfg,
        study.best_trial.params.get("hidden_dim", base_cfg["model"]["hidden_dim"]),
        study.best_trial.params.get("lr", base_cfg["training"]["lr"]),
        study.best_trial.params.get("dropout_p", None),
        study.best_trial.params.get("threshold", None),
    )
    out_path = Path(args.save_best_config) if args.save_best_config else Path(args.config).with_suffix(".tuned.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)
    print(f"\nSaved tuned config to: {out_path}")

    # Reproduce best
    cmd = f"{sys.executable} {script_path} --config {out_path}"
    if args.epochs is not None:
        cmd += f" --epochs {args.epochs}"
    if args.batch_size is not None:
        cmd += f" --batch_size {args.batch_size}"
    if args.max_train_batches is not None:
        cmd += f" --max_train_batches {args.max_train_batches} --debug"
    print("\nReproduce best:")
    print(cmd)

if __name__ == "__main__":
    main()

