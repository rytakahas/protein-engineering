
#!/usr/bin/env python3
import argparse, json, os, numpy as np, pathlib, time

def psi_value(p, q, eps=1e-6):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum((p - q) * np.log(p / q)))

def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

def load_baseline(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_hist(probs: np.ndarray, edges: np.ndarray):
    hist, _ = np.histogram(probs, bins=edges)
    total = hist.sum()
    if total == 0:
        return np.zeros_like(hist, dtype=np.float64)
    return hist.astype(np.float64) / float(total)

def main():
    ap = argparse.ArgumentParser(description="Batch PSI monitor for evaluation splits.")
    ap.add_argument("--baseline", required=True, help="baseline.json (edges + proportions from train)")
    ap.add_argument("--scores-npy", required=True, help="(N,) flattened probabilities from eval (masked upper triangle)")
    ap.add_argument("--outdir", default="reports", help="output directory for json")
    ap.add_argument("--split", default="val", help="split name for filenames")
    args = ap.parse_args()

    base = load_baseline(args.baseline)
    edges = np.array(base["edges"], dtype=np.float64)
    p = np.array(base["proportions"], dtype=np.float64)
    q = compute_hist(np.load(args.scores_npy), edges)
    val = psi_value(p, q)

    meta = dict(split=args.split, ts=timestamp(), baseline=os.path.basename(args.baseline))
    out = dict(psi=val, category=("stable" if val<=0.10 else "watch" if val<=0.25 else "drift"),
               proportions=list(map(float, q)), meta=meta)
    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)
    js = os.path.join(args.outdir, f"psi_{args.split}_{meta['ts']}.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[monitor_eval] PSI={val:.4f} → {js}")

if __name__ == "__main__":
    main()
