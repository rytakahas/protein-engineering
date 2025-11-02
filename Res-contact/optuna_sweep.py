#!/usr/bin/env python3
import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import yaml
import optuna

# Matches your printed line, e.g.:
# [epoch 5] train=0.1154  val=0.1147  ROC=0.840  PR=0.576  F1=0.526  (t=0.50)  BF1=0.605@0.30  (67.6s)
EPOCH_LINE_BF1 = re.compile(
    r"^\[epoch\s+(\d+)\]\s+train=([0-9.]+)\s+val=([0-9.]+)\s+ROC=([0-9.]+|na)\s+PR=([0-9.]+|na)\s+F1=([0-9.]+|na)\s+\(t=([0-9.]+)\)\s+BF1=([0-9.]+|na)@([0-9.]+|na)"
)
# Fallback (older format without BF1)
EPOCH_LINE_SIMPLE = re.compile(
    r"^\[epoch\s+(\d+)\]\s+train=([0-9.]+)\s+val=([0-9.]+)\s+ROC=([0-9.]+|na)\s+PR=([0-9.]+|na)\s+F1=([0-9.]+|na)"
)

def _float_or_nan(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _choose_objective_value(objective: str, val_loss: float, pr: float, f1: float, bf1: float) -> float:
    # Note: larger is better for all except neg_val (we use negative val loss so larger is better).
    if objective == "bf1":
        return bf1
    if objective == "f1":
        return f1
    if objective == "pr":
        return pr
    if objective == "neg_val":
        return -val_loss
    # default
    return bf1

def _make_pruner(name: str, warmup: int):
    name = (name or "median").lower()
    if name == "none":
        return optuna.pruners.NopPruner()
    if name == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=warmup)
    if name == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3, min_early_stopping_rate=0)
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    return optuna.pruners.MedianPruner(n_warmup_steps=warmup)

def _write_log(log_dir: Path, trial_number: int, buf: str) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"trial_{trial_number:03d}_{stamp}.log"
    log_path.write_text(buf)
    return str(log_path)

def run_one_trial_stream(
    trial: optuna.Trial,
    trial_cfg: dict,
    script_path: Path,
    extra_args,
    log_dir: Path,
    objective: str,
    prune_warmup_epochs: int,
) -> Tuple[float, str, str, str]:
    # Write a temp YAML config for this trial
    with tempfile.TemporaryDirectory() as td:
        tmp_cfg_path = Path(td) / "trial.yaml"
        tmp_cfg_path.write_text(yaml.safe_dump(trial_cfg, sort_keys=False))

        cmd = [sys.executable, str(script_path), "--config", str(tmp_cfg_path)]
        cmd += extra_args

        # Stream stdout line-by-line so we can report per-epoch metrics & prune.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        best_metric = -float("inf")
        best_kind = objective
        seen_epochs = 0
        buffer_lines = []

        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                buffer_lines.append(raw_line)

                m = EPOCH_LINE_BF1.match(line) or EPOCH_LINE_SIMPLE.match(line)
                if m:
                    seen_epochs += 1
                    # Parse common positions
                    # [epoch, train, val, roc, pr, f1, (maybe t), (maybe bf1), (maybe bf1_tau)]
                    groups = m.groups()
                    # indices are safe across both regexes for val/pr/f1
                    val_loss = _float_or_nan(groups[2])
                    pr_val  = _float_or_nan(groups[4])
                    f1_val  = _float_or_nan(groups[5])
                    # BF1 exists only in the BF1 regex; default to NaN otherwise
                    bf1_val = float("nan")
                    if m.re is EPOCH_LINE_BF1:
                        bf1_val = _float_or_nan(groups[7])

                    current = _choose_objective_value(objective, val_loss, pr_val, f1_val, bf1_val)

                    # Track best so far
                    if not (current != current):  # not NaN
                        best_metric = max(best_metric, current)

                    # Report to Optuna so pruners can act
                    trial.report(current if not (current != current) else -float("inf"), step=seen_epochs)

                    # Ask pruner after warmup
                    if seen_epochs >= prune_warmup_epochs and trial.should_prune():
                        # kill child proc
                        try:
                            proc.terminate()
                            # give it a moment
                            time.sleep(0.5)
                            if proc.poll() is None:
                                proc.kill()
                        except Exception:
                            pass
                        log_text = "".join(buffer_lines)
                        log_path = _write_log(log_dir, trial.number, log_text)
                        raise optuna.TrialPruned(f"Pruned at epoch {seen_epochs}. Log: {log_path}")

            # Process finished; collect remainder
            proc.wait()

            log_text = "".join(buffer_lines)
            log_path = _write_log(log_dir, trial.number, log_text)

            if proc.returncode != 0:
                return -float("inf"), "fail", log_text, log_path

            # If we never matched an epoch line, return -inf
            if best_metric == -float("inf"):
                return -float("inf"), "none", log_text, log_path

            return best_metric, objective, log_text, log_path

        finally:
            # Ensure the process is gone
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass


def make_trial_config(
    base_cfg: dict,
    hidden_dim: int,
    lr: float,
    maybe_dropout: Optional[float],
    maybe_thresh: Optional[float],
):
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # deep copy via yaml
    cfg["model"]["hidden_dim"] = int(hidden_dim)
    cfg["training"]["lr"] = float(lr)
    if maybe_dropout is not None:
        cfg["model"]["dropout_p"] = float(maybe_dropout)  # train.py reads this
    if maybe_thresh is not None:
        cfg.setdefault("inference", {})["threshold"] = float(maybe_thresh)
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Optuna sweep for rescontact (streaming, prunable).")
    ap.add_argument("--config", default="configs/rescontact.yaml", help="Base YAML config.")
    ap.add_argument("--script", default="scripts/train.py", help="Training entry script.")
    ap.add_argument("--trials", type=int, default=20, help="Number of trials.")
    ap.add_argument("--timeout", type=int, default=None, help="Global timeout (seconds).")
    ap.add_argument("--study", default=None, help="Optuna storage URL (e.g., sqlite:///rescontact.db).")
    ap.add_argument("--study-name", default="rescontact-sweep", help="Study name.")
    ap.add_argument("--seed", type=int, default=42)

    # Laptop-speed knobs forwarded to train.py
    ap.add_argument("--epochs", type=int, default=6, help="Override epochs per trial.")
    ap.add_argument("--max-train-batches", type=int, default=60, help="Cap batches/epoch.")
    ap.add_argument("--batch-size", type=int, default=1)

    # Convenience knobs → forwarded to train.py
    ap.add_argument("--min-train-examples", type=int, default=None, help="Alias of train.py --max_train_examples.")
    ap.add_argument("--val-split", type=float, default=None, help="Alias of train.py --train_val_split.")

    # Search spaces
    ap.add_argument("--tune-hidden", action="store_true", default=True)
    ap.add_argument("--tune-lr", action="store_true", default=True)
    ap.add_argument("--tune-dropout", action="store_true", default=False)
    ap.add_argument("--tune-threshold", action="store_true", default=False)

    ap.add_argument("--space-hidden", nargs="+", type=int, default=[64, 96, 128, 256])
    ap.add_argument("--space-lr", nargs="+", type=float, default=[1e-3, 1.5e-3, 5e-4])
    ap.add_argument("--space-dropout", nargs="+", type=float, default=[0.0, 0.1, 0.2])

    # Widened defaults (you can override on CLI)
    ap.add_argument("--thresh-min", type=float, default=0.10)
    ap.add_argument("--thresh-max", type=float, default=0.60)
    ap.add_argument("--thresh-step", type=float, default=0.02)

    # Objective & pruning
    ap.add_argument("--objective", choices=["bf1", "f1", "pr", "neg_val"], default="bf1",
                    help="Optimize this metric parsed per-epoch from logs.")
    ap.add_argument("--pruner", choices=["none", "median", "successive_halving", "hyperband"], default="median")
    ap.add_argument("--prune-warmup-epochs", type=int, default=2,
                    help="Do not prune before this many epochs.")

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

    # Prepare study (with pruner)
    pruner = _make_pruner(args.pruner, args.prune_warmup_epochs)
    if args.study:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.study,
            direction="maximize",
            load_if_exists=True,
            pruner=pruner,
        )
    else:
        study = optuna.create_study(direction="maximize", pruner=pruner)

    # Extra CLI args forwarded to train.py
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

    # Build threshold grid if requested
    def _threshold_grid():
        v = args.thresh_min
        grid = []
        while v <= args.thresh_max + 1e-9:
            grid.append(round(v, 3))
            v += args.thresh_step
        return grid

    def objective(trial: optuna.Trial):
        # sample hyperparams
        hidden_dim = trial.suggest_categorical("hidden_dim", args.space_hidden) if args.tune_hidden else base_cfg["model"]["hidden_dim"]
        lr = trial.suggest_categorical("lr", args.space_lr) if args.tune_lr else base_cfg["training"]["lr"]
        dropout = trial.suggest_categorical("dropout_p", args.space_dropout) if args.tune_dropout else None

        thresh = None
        if args.tune_threshold:
            thresh = trial.suggest_categorical("threshold", _threshold_grid())

        trial_cfg = make_trial_config(base_cfg, hidden_dim, lr, dropout, thresh)

        try:
            metric, kind, stdout, log_path = run_one_trial_stream(
                trial=trial,
                trial_cfg=trial_cfg,
                script_path=script_path,
                extra_args=extra_args,
                log_dir=log_dir,
                objective=args.objective,
                prune_warmup_epochs=args.prune_warmup_epochs,
            )
        except optuna.TrialPruned as e:
            # Attach last lines for convenience
            trial.set_user_attr("metric_kind", f"{args.objective}(pruned)")
            trial.set_user_attr("stdout_tail", str(e))
            raise

        trial.set_user_attr("metric_kind", kind)
        trial.set_user_attr("log_path", log_path)
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

