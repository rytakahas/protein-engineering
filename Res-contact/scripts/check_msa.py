#!/usr/bin/env python3
import time
from time import perf_counter
import yaml
import torch
from pathlib import Path
import sys
import argparse
from typing import List, Optional

# --- resolve project root (.. from /scripts) and add ./src to sys.path ---
ROOT = Path(__file__).resolve().parents[1]  # <-- parent of /scripts
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# ------------------------------------------------------------------------

from rescontact.data.dataset import PDBContactDataset  # type: ignore


def parse_args():
    ap = argparse.ArgumentParser(description="Quick MSA sanity checker (ESM+MSA dims).")
    ap.add_argument("--config", default="configs/rescontact.yaml", help="Path to YAML config.")
    ap.add_argument("--n", type=int, default=3, help="How many examples to probe.")
    ap.add_argument("--providers", type=str, default=None,
                    help="Override MSA provider order, comma-separated (e.g. 'local,jackhmmer_remote').")
    ap.add_argument("--no-msa", action="store_true", help="Force disable MSA (ESM-only).")
    ap.add_argument("--email", type=str, default=None,
                    help="jackhmmer_remote email (optional but recommended).")
    ap.add_argument("--timeout", type=int, default=None,
                    help="jackhmmer_remote timeout_s override (seconds).")
    return ap.parse_args()


def override_providers(msa_cfg: dict, providers_csv: Optional[str]) -> None:
    if not providers_csv:
        return
    provs: List[str] = [p.strip() for p in providers_csv.split(",") if p.strip()]
    msa_cfg["provider_order"] = provs


def to_np(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t


def main():
    args = parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.no_msa:
        cfg["features"]["use_msa"] = False
        print("[check_msa] Forcing ESM-only (use_msa=false).")

    msa_cfg = cfg["features"]["msa"]
    override_providers(msa_cfg, args.providers)

    # Optional jackhmmer_remote tweaks
    if args.email:
        msa_cfg.setdefault("jackhmmer_remote", {})["email"] = args.email
    if args.timeout is not None:
        msa_cfg.setdefault("jackhmmer_remote", {})["timeout_s"] = int(args.timeout)

    print("[check_msa] effective provider_order:", msa_cfg.get("provider_order"))

    ds = PDBContactDataset(
        root_dir=cfg["paths"]["train_dir"],
        cache_dir=cfg["paths"]["cache_dir"],
        contact_threshold=cfg["labels"]["contact_threshold_angstrom"],
        include_inter_chain=cfg["labels"]["include_inter_chain"],
        esm_model_name=cfg["model"]["esm_model"],
        use_msa=cfg["features"]["use_msa"],
        msa_cfg=msa_cfg,
    )

    print(f"dataset size: {len(ds)}")
    if len(ds) == 0:
        raise SystemExit("No examples found. Check paths.train_dir in configs/rescontact.yaml.")

    # Warm-up one example
    t0 = time.time()
    it0 = ds[0]
    dt0 = time.time() - t0
    print(f"loaded first example in {dt0:.2f}s; id={it0.get('id','?')}")

    emb = to_np(it0["emb"])
    L, D = emb.shape
    print("emb shape:", (L, D))
    print("msa_path:", it0.get("msa_path"))

    if D >= 341:
        last21 = emb[:, -21:]
        nonzero = int((last21 != 0).sum())
        print("nonzero in last-21 dims:", nonzero,
              "(>0 means BLAST/Jackhmmer/local MSA present; 0 means zero-padded)")
    else:
        print("No MSA concatenated (D < 341). If you expect MSA, set features.use_msa: true "
              "and ensure your dataset pads zeros when providers are unavailable).")

    # Probe first N items with timings
    N = min(args.n, len(ds))
    for i in range(N):
        t0 = perf_counter()
        itm = ds[i]
        dt = perf_counter() - t0
        emb_i = to_np(itm["emb"])
        Di = emb_i.shape[1]
        nz21 = int((emb_i[:, -21:] != 0).sum()) if Di >= 341 else -1
        prov = itm.get("msa_provider")
        if prov is None:
            prov = "local" if itm.get("msa_path") else ("remote/unknown" if (Di >= 341 and nz21 > 0) else "none")
        print(f"[{i}] id={itm.get('id','?')}  load={dt:.2f}s  D={Di}  "
              f"msa_nonzero={nz21}  msa_path={itm.get('msa_path')}  provider={prov}")


if __name__ == "__main__":
    main()

