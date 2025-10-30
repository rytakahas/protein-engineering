import os
import io
import base64
import yaml
import numpy as np
import torch
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel

from rescontact.models.contact_net import BilinearContactNet
from rescontact.features.embedding import ESMEmbedder


class PredictRequest(BaseModel):
    sequence: str | None = None
    pdb_path: str | None = None
    threshold: float | None = 0.5


def build_app(config_path: str = "configs/rescontact.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)

    model = BilinearContactNet(
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        distance_bias_max=cfg["model"]["distance_bias_max"],
    ).to(device)

    ckpt_path = os.path.join("artifacts", "rescontact_best.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    model.eval()

    embedder = ESMEmbedder(cfg["model"]["esm_model"], cfg["paths"]["cache_dir"], str(device))

    api = FastAPI()

    @api.post("/predict")
    def predict(req: PredictRequest):
        if not req.sequence and not req.pdb_path:
            return {"error": "Provide 'sequence' or 'pdb_path'."}
        if req.sequence:
            seq = req.sequence.strip()
        else:
            # derive sequence from structure
            from rescontact.data.pdb_utils import load_structure, chains_from_structure, chain_sequence
            s = load_structure(req.pdb_path)
            chains = chains_from_structure(s)
            seq = "".join(chain_sequence(chains[c]) for c in sorted(chains.keys()))
        with torch.no_grad():
            emb = torch.from_numpy(embedder.embed(seq)).to(device)
            logits = model(emb)
            probs = torch.sigmoid(logits).cpu().numpy()
        thr = float(req.threshold if req.threshold is not None else cfg["inference"]["threshold"])
        binary = (probs >= thr).astype(np.uint8)
        buf = io.BytesIO()
        np.savez_compressed(buf, probs=probs, binary=binary)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"length": int(probs.shape[0]), "npz_b64": b64}

    @api.post("/train")
    def train_endpoint():
        """Blocking training call that shells out to scripts/train.py with this config."""
        try:
            subprocess.run(["python", "scripts/train.py", "--config", config_path], check=True)
            return {"status": "ok"}
        except subprocess.CalledProcessError as e:
            return {"status": "error", "detail": str(e)}

    return api


app = build_app()

