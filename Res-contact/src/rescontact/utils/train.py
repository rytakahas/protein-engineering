import os
import random
import torch
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference=("mps", "cpu")):
    for dev in preference:
        if dev == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if dev == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if dev == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


def save_checkpoint(model: torch.nn.Module, path: str, cfg: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)


def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt.get("config", {})


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.wait = 0

    def step(self, val):
        if val < self.best - self.min_delta:
            self.best = val
            self.wait = 0
            return False
        self.wait += 1
        return self.wait > self.patience

