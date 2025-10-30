# src/rescontact/data/dataset.py
import os, math, hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from Bio.PDB import PDBParser, MMCIFParser
from Bio.Data.IUPACData import protein_letters_3to1

# -----------------------------
# Utilities
# -----------------------------

_AA3_TO_1_FALLBACK = {
    "SEC": "C", "PYL": "K",
    "ASX": "B", "GLX": "Z", "XLE": "J",
    "XAA": "X", "UNK": "X",
}

def _three_to_one_safe(res3: str) -> str:
    r = (res3 or "").upper().strip()
    if r in protein_letters_3to1:
        return protein_letters_3to1[r]
    return _AA3_TO_1_FALLBACK.get(r, "X")

def _residue_one_letter(resname: str) -> str:
    try:
        return _three_to_one_safe(resname)
    except Exception:
        return "X"

def contact_map_from_coords(
    coords: np.ndarray,
    threshold_angstrom: float = 8.0,
    sym: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build binary contact map (Cβ–Cβ; Cα for Gly) by threshold on Euclidean distance.
    Returns (Y, M) as torch.float32 with shape [L,L], diagonal masked to 0.

    coords: [L,3] np.ndarray
    """
    L = int(coords.shape[0])
    if L == 0:
        z = torch.zeros((0, 0), dtype=torch.float32)
        return z, z

    diffs = coords[:, None, :] - coords[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    dist = np.sqrt(np.maximum(d2, 0.0))
    Y = (dist <= float(threshold_angstrom)).astype(np.float32)
    np.fill_diagonal(Y, 0.0)  # no self-contacts

    if sym:
        Y = np.maximum(Y, Y.T)

    M = np.ones_like(Y, dtype=np.float32)
    np.fill_diagonal(M, 0.0)

    return torch.from_numpy(Y), torch.from_numpy(M)


def _hash_string(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


# -----------------------------
# ESM2 embedding (Hugging Face)
# -----------------------------
class _HFESM2:
    """
    Minimal HF wrapper for ESM2. Produces per-residue embeddings [L, 320].
    Cached on disk to speed up repeated runs.
    """
    def __init__(self, model_id: str, cache_dir: Path, device: torch.device, verbose: int = 1):
        from transformers import AutoTokenizer, AutoModel

        self.model_id = model_id
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.verbose = verbose

        if self.verbose:
            print(f"[rescontact/ESM] init model_id={model_id} cache={str(cache_dir)} device={device.type}")

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()

    def _cache_path(self, seq: str) -> Path:
        hid = _hash_string(self.model_id + "::" + seq)
        return self.cache_dir / f"{hid}.npy"

    @torch.no_grad()
    def embed_sequence(self, seq: str) -> torch.Tensor:
        """
        Return [L, 320] torch.float32 on CPU.
        Uses disk cache under cache_dir.
        """
        cp = self._cache_path(seq)
        if cp.exists():
            arr = np.load(cp)
            return torch.from_numpy(arr.astype(np.float32))

        # Tokenize — ESM2 tokenizer splits amino-acid chars correctly
        enc = self.tok(
            seq,
            return_tensors="pt",
            add_special_tokens=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.model(**enc)
        hs = out.last_hidden_state  # [1, L+2, 320] (CLS + seq + EOS)
        # Drop special tokens: keep residues only (1..L)
        L = len(seq)
        reps = hs[:, 1 : 1 + L, :].squeeze(0)  # [L, 320]
        reps_cpu = reps.detach().to("cpu").float().contiguous()

        # Cache
        np.save(cp, reps_cpu.numpy())
        return reps_cpu


# -----------------------------
# PDB / mmCIF parsing
# -----------------------------
def _chain_sequence_and_coords(struct, chain_id: str) -> Tuple[str, np.ndarray]:
    """
    Extract one-letter sequence and Nx3 coordinates for a chain.
    Prefers Cβ, falls back to Cα. Skips residues without these atoms.
    """
    seq: List[str] = []
    coords: List[List[float]] = []

    model = next(struct.get_models())

    # Find requested chain; if absent, use first
    chain = None
    for ch in model.get_chains():
        if ch.id == chain_id:
            chain = ch
            break
    if chain is None:
        chain = next(model.get_chains())

    for res in chain.get_residues():
        hetflag, resseq, icode = res.get_id()
        if hetflag != " ":
            continue  # skip HETATM/waters

        resname = res.get_resname()
        aa = _residue_one_letter(resname)

        xyz = None
        if res.has_id("CB"):
            xyz = res["CB"].get_coord()
        elif res.has_id("CA"):
            xyz = res["CA"].get_coord()
        else:
            continue

        seq.append(aa)
        coords.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])

    if not seq:
        return "", np.zeros((0, 3), dtype=np.float32)

    return "".join(seq), np.asarray(coords, dtype=np.float32)


def _load_structure(path: Path):
    if path.suffix.lower() in [".cif", ".mmcif"]:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(id=path.stem, file=str(path))


# -----------------------------
# Dataset
# -----------------------------
class PDBContactDataset(Dataset):
    """
    Returns dict:
      {
        "id": str,
        "emb": torch.FloatTensor [L, D],
        "contacts": torch.FloatTensor [L, L],
        "mask": torch.FloatTensor [L, L],
        "msa_path": Optional[str]   # None/"" in ESM-only or zero-padded MSA
      }
    """

    def __init__(
        self,
        root_dir: str,
        cache_dir: str,
        contact_threshold: float = 8.0,
        include_inter_chain: bool = True,   # currently not used (single chain)
        esm_model_name: str = "facebook/esm2_t6_8M_UR50D",
        use_msa: bool = False,
        msa_cfg: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        list_first_n: Optional[int] = None,
    ):
        self.root = Path(root_dir)
        self.cache_root = Path(cache_dir)
        # Avoid ".../emb/emb"
        self.emb_cache = self.cache_root / "emb" if self.cache_root.name != "emb" else self.cache_root
        self.emb_cache.mkdir(parents=True, exist_ok=True)

        self.contact_threshold = float(contact_threshold)
        self.include_inter_chain = bool(include_inter_chain)
        self.esm_model_name = esm_model_name
        self.use_msa = bool(use_msa)
        self.msa_cfg = msa_cfg or {}

        self.device = device or (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
        self.verbose = int(os.getenv("RESCONTACT_VERBOSE", "1"))

        self._esm_backend: Optional[_HFESM2] = None

        self.items: List[Tuple[Path, str]] = self._enumerate_examples(self.root)
        if list_first_n is not None:
            self.items = self.items[: int(list_first_n)]

        if self.verbose:
            print(f"[rescontact/ds] enumerated {len(self.items)} examples under {self.root}")

    # ------------- file listing -------------
    def _enumerate_examples(self, root: Path) -> List[Tuple[Path, str]]:
        exts = (".pdb", ".ent", ".cif", ".mmcif")
        paths = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(p)
        # default to chain "A" (we’ll fallback to first found if missing)
        return [(p, "A") for p in sorted(paths)]

    # ------------- ESM backend -------------
    def _esm(self) -> _HFESM2:
        if self._esm_backend is None:
            self._esm_backend = _HFESM2(self.esm_model_name, self.emb_cache, self.device, verbose=self.verbose)
        return self._esm_backend

    def _esm_embed(self, seq: str) -> torch.Tensor:
        """
        Returns [L, 320] float32 CPU tensor.
        """
        return self._esm().embed_sequence(seq)

    # ------------- MSA 1-D features -------------
    def _msa_1d(self, seq: str) -> Tuple[torch.Tensor, Optional[str], Optional[str]]:
        """
        Return ([L,21] float32 CPU), msa_path, provider.
        This default impl returns zeros (no remote calls), but keeps shapes stable.
        """
        L = len(seq)
        msa_1d = torch.zeros((L, 21), dtype=torch.float32)
        return msa_1d, None, None

    # ------------- loader helpers -------------
    def _load_seq_coords(self, path: Path, chain_id: str) -> Tuple[str, np.ndarray]:
        struct = _load_structure(path)
        return _chain_sequence_and_coords(struct, chain_id)

    # ------------- PyTorch API -------------
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        path, chain_id = self.items[idx]
        seq, coords = self._load_seq_coords(path, chain_id)
        if len(seq) == 0 or coords.shape[0] == 0 or coords.shape[0] != len(seq):
            # bad structure; let the collate_fn skip it
            if self.verbose:
                print(f"[rescontact/ds] skip malformed {path.name} (seq={len(seq)} coords={coords.shape})")
            return None

        # ESM per-residue embeddings [L,320]
        try:
            emb_esm = self._esm_embed(seq)  # [L,320], CPU
        except Exception as e:
            if self.verbose:
                print(f"[rescontact/ESM] embedding failed on {path.name}: {e}")
            return None

        # Optional MSA 1-D features [L,21]
        msa_path: Optional[str] = None
        if self.use_msa:
            msa_1d, msa_path, _ = self._msa_1d(seq)
            emb = torch.cat([emb_esm, msa_1d], dim=-1)  # [L,341]
        else:
            emb = emb_esm  # [L,320]

        # Contacts & mask
        Y, M = contact_map_from_coords(coords, threshold_angstrom=self.contact_threshold, sym=True)

        return {
            "id": f"{path.name}::{chain_id}",
            "emb": emb,                 # [L, D]
            "contacts": Y,              # [L, L]
            "mask": M,                  # [L, L]
            "msa_path": msa_path,       # None or "" when not used
        }

